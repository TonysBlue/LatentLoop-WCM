from __future__ import annotations

import json
import os
import pickle
import subprocess
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from omegaconf import OmegaConf
from runtime.config import ProjectConfig, load_config

from training.checkpoint import config_hash, file_sha256
from training.evaluation import build_evaluation_report, evaluate_checkpoint
from training.training import train

_FORMAL_STAGES = ("pretrain", "sft", "rl")
_THREE_STAGE_DATASETS = {"synthetic", "canary"}


@dataclass(frozen=True, slots=True)
class RecipeStage:
    name: str
    config: str


@dataclass(frozen=True, slots=True)
class TrainingRecipe:
    name: str
    dataset: str
    stages: tuple[RecipeStage, ...]
    initial_checkpoint: str | None = None
    run_id: str | None = None
    validation_after_each_stage: bool = True
    test_after_final_stage: bool = True


def load_recipe(path: str | Path) -> TrainingRecipe:
    path = Path(path).expanduser().resolve()
    raw = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
    if not isinstance(raw, dict):
        raise ValueError("recipe must be a mapping")
    stages_raw = raw.get("stages")
    if not isinstance(stages_raw, list) or not stages_raw:
        raise ValueError("recipe must contain at least one stage")
    stages = tuple(
        RecipeStage(name=str(item["name"]), config=str(item["config"])) for item in stages_raw
    )
    if len({stage.name for stage in stages}) != len(stages):
        raise ValueError("recipe stage names must be unique")
    return TrainingRecipe(
        name=str(raw["name"]),
        dataset=str(raw["dataset"]),
        stages=stages,
        initial_checkpoint=raw.get("initial_checkpoint"),
        run_id=str(raw["run_id"]) if raw.get("run_id") else None,
        validation_after_each_stage=bool(
            raw.get("evaluation", {}).get("validation_after_each_stage", True)
        ),
        test_after_final_stage=bool(raw.get("evaluation", {}).get("test_after_final_stage", True)),
    )


def _resolve_config(recipe_path: Path, stage: RecipeStage) -> Path:
    candidate = Path(stage.config).expanduser()
    if not candidate.is_absolute():
        candidate = (recipe_path.parent / candidate).resolve()
    if not candidate.is_file():
        raise FileNotFoundError(f"stage config is absent: {candidate}")
    return candidate


def _checkpoint_path(config: ProjectConfig, updates: int) -> Path:
    return config.runtime.root_path() / "checkpoints" / f"step-{updates:08d}.pt"


def _latest_checkpoint(config: ProjectConfig) -> Path | None:
    checkpoints = sorted(config.runtime.root_path().glob("checkpoints/step-*.pt"))
    return checkpoints[-1] if checkpoints else None


def _new_run_id() -> str:
    """Create a unique default while allowing explicit IDs for reproducibility."""
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return f"{stamp}-{uuid4().hex[:8]}"


def _experiment_base(config: ProjectConfig) -> Path:
    value = os.environ.get("LATENTLOOP_EXPERIMENT_ROOT")
    if value:
        return Path(value).expanduser().resolve()
    # Config profiles keep the storage layout under <storage>/experiments/<profile>/<run>.
    return config.runtime.data_path().parent / "experiments"


def _checkpoint_matches(path: Path, config: ProjectConfig) -> bool:
    try:
        import torch

        payload = torch.load(path, map_location="cpu", weights_only=False)
        return (
            payload.get("config_hash") == config_hash(config.as_dict())
            and payload.get("metadata", {}).get("data_identity") == _data_identity(config)
            and payload.get("metadata", {}).get("codec_id") == config.data.codec_id
            and payload.get("metadata", {}).get("codec_revision") == config.data.codec_revision
            and payload.get("metadata", {}).get("codec_weight_hash")
            == config.data.codec_weight_hash
            and payload.get("metadata", {}).get("stage") == config.training.stage
            and payload.get("metadata", {}).get("algorithm")
            == (
                config.training.rl.algorithm
                if config.training.stage == "rl"
                else None
            )
            and "objective" not in payload.get("metadata", {})
        )
    except (OSError, RuntimeError, ValueError, KeyError, TypeError, pickle.UnpicklingError):
        return False


def _data_identity(config: ProjectConfig) -> str:
    if config.data.manifest:
        manifest = Path(config.data.manifest).expanduser()
        if not manifest.is_absolute():
            manifest = config.runtime.data_path() / manifest
        if manifest.is_file():
            return file_sha256(manifest)
    return config_hash(config.as_dict()["data"])


def _checkpoint_metadata(path: Path) -> dict[str, Any]:
    import torch

    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as error:
        return {"error": str(error)}
    return {
        "config_hash": payload.get("config_hash"),
        "metadata": payload.get("metadata", {}),
        "train_state": payload.get("train_state", {}),
    }


def _require_parent_stage(path: Path, *, stage: str) -> None:
    import torch

    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except (OSError, RuntimeError, ValueError, TypeError) as error:
        raise ValueError(f"cannot inspect parent checkpoint {path}: {error}") from error
    if not isinstance(payload.get("model"), dict) or not isinstance(
        payload.get("metadata"), dict
    ):
        raise ValueError("formal stage parent checkpoint is incomplete")
    metadata = payload.get("metadata", {})
    if (
        metadata.get("stage") != stage
        or metadata.get("algorithm") is not None
        or "objective" in metadata
    ):
        raise ValueError(
            f"formal stage requires a current supervised {stage} checkpoint, got "
            f"stage={metadata.get('stage')}, algorithm={metadata.get('algorithm')}"
        )


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return "nogit"


def _manifest_units(config: ProjectConfig) -> int:
    if not config.data.manifest:
        return 0
    path = Path(config.data.manifest).expanduser()
    if not path.is_absolute():
        path = config.runtime.data_path() / path
    if not path.is_file():
        return 0
    with path.open(encoding="utf-8") as source:
        total = 0
        for line in source:
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            total += int(record.get("units", record.get("unit_count", 0)))
        return total


def run_recipe(
    path: str | Path,
    overrides: list[str] | None = None,
    *,
    run_id: str | None = None,
) -> dict[str, Any]:
    recipe_path = Path(path).expanduser().resolve()
    recipe = load_recipe(recipe_path)
    if recipe.dataset in _THREE_STAGE_DATASETS:
        if tuple(stage.name for stage in recipe.stages) != _FORMAL_STAGES:
            raise ValueError("three-stage recipes must contain pretrain -> sft -> rl")
    selected_run_id = (
        run_id or recipe.run_id or os.environ.get("LATENTLOOP_RUN_ID") or _new_run_id()
    )
    if not selected_run_id.replace("-", "").replace("_", "").isalnum():
        raise ValueError("run_id may contain only letters, numbers, hyphens and underscores")
    if recipe.dataset not in _THREE_STAGE_DATASETS | {"direct-speech-overfit"}:
        raise ValueError(f"recipe dataset is not supported: {recipe.dataset}")
    parent = Path(recipe.initial_checkpoint).expanduser() if recipe.initial_checkpoint else None
    reports: list[dict[str, Any]] = []
    for index, stage in enumerate(recipe.stages):
        config_path = _resolve_config(recipe_path, stage)
        config = load_config(config_path, overrides)
        if config.data.dataset != recipe.dataset:
            raise ValueError(
                f"stage {stage.name} dataset {config.data.dataset!r} differs from "
                f"recipe {recipe.dataset!r}"
            )
        expected_stage = (
            stage.name if recipe.dataset in _THREE_STAGE_DATASETS else config.training.stage
        )
        if config.training.stage != expected_stage:
            raise ValueError(
                f"stage {stage.name} must use training.stage={expected_stage}, got "
                f"{config.training.stage}"
            )
        config.runtime.run_name = f"{recipe.name}-{stage.name}"
        config.runtime.recipe_name = recipe.name
        config.runtime.stage_name = stage.name
        config.runtime.experiment_root = str(
            _experiment_base(config) / recipe.name / selected_run_id / stage.name
        )
        config.runtime.root_path().mkdir(parents=True, exist_ok=True)
        stage_dir = config.runtime.root_path()
        final_checkpoint = _checkpoint_path(config, config.training.max_updates)
        latest = _latest_checkpoint(config)
        if latest and not _checkpoint_matches(latest, config):
            metadata = _checkpoint_metadata(latest)
            raise ValueError(
                f"checkpoint {latest} does not match stage {stage.name} configuration/data "
                f"(config_hash={metadata.get('config_hash')})"
            )
        if final_checkpoint.is_file() and not _checkpoint_matches(final_checkpoint, config):
            metadata = _checkpoint_metadata(final_checkpoint)
            raise ValueError(
                f"completed checkpoint {final_checkpoint} does not match stage {stage.name} "
                f"configuration/data (config_hash={metadata.get('config_hash')})"
            )
        resume = str(latest) if latest and latest != final_checkpoint else None
        if final_checkpoint.is_file():
            completed: dict[str, Any] = {
                "recipe": recipe.name,
                "dataset": recipe.dataset,
                "stage": stage.name,
                "run_id": selected_run_id,
                "stage_index": index,
                "config": str(config_path),
                "checkpoint": str(final_checkpoint),
                "status": "already-complete",
            }
            if recipe.validation_after_each_stage:
                completed["validation"] = _evaluate_and_write(
                    config,
                    final_checkpoint,
                    "validation",
                    stage_dir / "reports" / "validation.json",
                )
            reports.append(completed)
            parent = final_checkpoint
            continue
        init_from = None if resume else (str(parent) if parent else None)
        if recipe.dataset in _THREE_STAGE_DATASETS and init_from:
            required_parent = "pretrain" if stage.name == "sft" else "sft"
            _require_parent_stage(Path(init_from), stage=required_parent)
        result = train(config, resume=resume, init_from=init_from)
        final_checkpoint = _checkpoint_path(config, result["train_state"]["update"])
        stage_report: dict[str, Any] = {
            "recipe": recipe.name,
            "dataset": recipe.dataset,
            "stage": stage.name,
            "run_id": selected_run_id,
            "stage_index": index,
            "config": str(config_path),
            "checkpoint": str(final_checkpoint),
            "train": {
                "train_state": result["train_state"],
                "metrics": result["metrics"],
                "tracking": result["tracking"],
            },
            "git_commit": _git_commit(),
            "progress": {
                "optimizer_updates": result["train_state"]["update"],
                "consumed_units": result["train_state"].get("consumed_units", 0),
                "estimated_epochs": (
                    result["train_state"].get("consumed_units", 0) / max(_manifest_units(config), 1)
                ),
            },
        }
        if recipe.validation_after_each_stage:
            stage_report["validation"] = _evaluate_and_write(
                config, final_checkpoint, "validation", stage_dir / "reports" / "validation.json"
            )
        reports.append(stage_report)
        parent = final_checkpoint
    if recipe.test_after_final_stage:
        final = reports[-1]
        config = load_config(_resolve_config(recipe_path, recipe.stages[-1]), overrides)
        final_checkpoint = Path(final["checkpoint"])
        final["test"] = _evaluate_and_write(
            config,
            final_checkpoint,
            "test",
            final_checkpoint.parent.parent / "reports" / "test.json",
        )
    output = {
        "recipe": recipe.name,
        "dataset": recipe.dataset,
        "run_id": selected_run_id,
        "stages": reports,
    }
    output_path = Path(reports[-1]["checkpoint"]).parent.parent / "recipe-report.json"
    output_path.write_text(json.dumps(output, indent=2, default=str) + "\n", encoding="utf-8")
    return output


def _evaluate_and_write(
    config: ProjectConfig, checkpoint: Path, split: str, report_path: Path
) -> dict[str, Any]:
    result = evaluate_checkpoint(config, checkpoint, split=split)
    payload = build_evaluation_report(config, checkpoint, split, result)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload
