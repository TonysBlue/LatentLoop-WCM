from __future__ import annotations

import glob
import shutil
from pathlib import Path
from typing import Any

import torch
from runtime.config import ProjectConfig

from data.curation.common import SPLITS, dataset_path, read_json, read_jsonl, write_json


def _required(path: Path, label: str, missing: list[str]) -> None:
    if not path.is_file():
        missing.append(f"{label}: {path}")


def _resolve_config_path(config: ProjectConfig, value: str | None) -> Path | None:
    if not value:
        return None
    path = Path(value).expanduser()
    return path if path.is_absolute() else config.runtime.data_path() / path


def _path_is_dataset_member(path: Path, dataset_root: Path) -> bool:
    try:
        path.resolve().relative_to(dataset_root.resolve())
        return True
    except ValueError:
        return False


def check_readiness(
    root: str | Path,
    *,
    config: ProjectConfig,
    dataset: str | None = None,
    require_checkpoint: str | Path | None = None,
    require_encoded: bool = True,
) -> dict[str, Any]:
    """Fail-closed machine check before any real-data training run."""
    dataset = dataset or config.data.dataset
    if dataset not in {"canary", "direct-speech-overfit"}:
        raise ValueError(f"readiness is not required for dataset={dataset!r}")
    root = Path(root).expanduser().resolve()
    if not root.is_dir():
        raise ValueError(f"training readiness failed: data root: {root}")
    dataset_root = (
        root / "gates" / "direct-speech-overfit"
        if dataset == "direct-speech-overfit"
        else dataset_path(root, dataset)
    )
    missing: list[str] = []
    invalid: list[str] = []
    formal_dataset = dataset == "canary"

    if formal_dataset and config.training.stage == "rl":
        guard_assets = (
            ("SFT replay manifest", config.training.rl.sft_replay_manifest, False),
            ("SFT replay shards", config.training.rl.sft_replay_shards, True),
            (
                "SFT preservation manifest",
                config.training.rl.sft_preservation_manifest,
                False,
            ),
            (
                "SFT preservation shards",
                config.training.rl.sft_preservation_shards,
                True,
            ),
        )
        for label, value, is_pattern in guard_assets:
            resolved = _resolve_config_path(config, value)
            if resolved is None:
                missing.append(f"{label}: unconfigured")
            elif is_pattern and not glob.glob(str(resolved)):
                missing.append(f"{label}: {resolved}")
            elif not is_pattern:
                _required(resolved, label, missing)

    def path(*parts: str) -> Path:
        return dataset_root.joinpath(*parts)

    audit_path = path("reports", "audit.json")
    if formal_dataset:
        _required(audit_path, "audit report", missing)
        if audit_path.is_file() and not read_json(audit_path).get("passed"):
            invalid.append(f"audit report is not passed: {audit_path}")
    manifest_paths: dict[str, Path] = {}
    shard_paths: dict[str, list[Path]] = {}
    splits = ("train",) if dataset == "direct-speech-overfit" else SPLITS
    for split in splits:
        manifest = path("manifests", f"{split}.jsonl")
        if dataset == "direct-speech-overfit" and not manifest.is_file():
            # The deterministic gate has no separate source manifest; its
            # processed manifest is the immutable source of episode identity.
            manifest = path("shards", "processed", split, f"{split}-manifest.jsonl")
        manifest_paths[split] = manifest
        _required(manifest, f"{split} manifest", missing)
        processed = path("shards", "processed", split)
        shards = sorted(processed.glob(f"{split}-*.tar"))
        shard_paths[split] = shards
        if split == "train":
            configured_manifest = _resolve_config_path(config, config.data.manifest)
            configured_shards = _resolve_config_path(config, config.data.shards)
            expected_processed_manifest = processed / "train-manifest.jsonl"
            if configured_manifest and not _path_is_dataset_member(
                configured_manifest, dataset_root
            ):
                invalid.append(
                    f"config data.manifest is outside {dataset} dataset root: {configured_manifest}"
                )
            if configured_manifest and configured_manifest.name != "train-manifest.jsonl":
                invalid.append(
                    "config data.manifest must be the processed train manifest: "
                    f"{configured_manifest}"
                )
            elif configured_manifest and (
                configured_manifest.resolve() != expected_processed_manifest.resolve()
            ):
                invalid.append(
                    "config data.manifest does not point to the processed train manifest: "
                    f"{configured_manifest}"
                )
            if configured_shards and not _path_is_dataset_member(configured_shards, dataset_root):
                invalid.append(
                    f"config data.shards is outside {dataset} dataset root: {configured_shards}"
                )
            expected_shard_root = dataset_root / "shards" / "processed" / "train"
            if configured_shards and not _path_is_dataset_member(
                configured_shards, expected_shard_root
            ):
                invalid.append(
                    f"config data.shards must target processed/train shards: {configured_shards}"
                )
        if require_encoded:
            if not shards:
                missing.append(f"{split} processed shards: {processed / (split + '-*.tar')}")
            _required(
                processed / f"{split}-manifest.jsonl",
                f"{split} processed manifest",
                missing,
            )
    if require_encoded:
        for split in splits:
            if not manifest_paths[split].is_file():
                continue
            source_entries = read_jsonl(manifest_paths[split])
            processed_manifest = path("shards", "processed", split, f"{split}-manifest.jsonl")
            if processed_manifest.is_file():
                encoded_entries = read_jsonl(processed_manifest)
                required_metadata = {
                    "protocol_version",
                    "action_schema_id",
                    "runtime_identity",
                    "environment_id",
                    "environment_version",
                }
                if any(required_metadata - set(entry) for entry in encoded_entries):
                    invalid.append(
                        f"{split} processed manifest is missing runtime identity"
                    )
                if {entry["episode_id"] for entry in source_entries} != {
                    entry["episode_id"] for entry in encoded_entries
                }:
                    invalid.append(f"{split} processed manifest does not match source manifest")
                if any(not entry.get("speech_codes_encoded") for entry in encoded_entries):
                    invalid.append(f"{split} processed manifest contains unencoded speech")
    mimi_reports = {}
    if formal_dataset:
        report = path("reports", "mimi-decode.json")
        _required(report, f"{dataset} Mimi decode report", missing)
        if report.is_file():
            mimi = read_json(report)
            mimi_reports[dataset] = mimi
            source_manifest = path("manifests", "episodes.jsonl")
            from data.curation.common import sha256_file

            if mimi.get("manifest_sha256") != (
                sha256_file(source_manifest) if source_manifest.is_file() else None
            ):
                invalid.append(
                    f"{dataset} Mimi report manifest hash does not match source manifest"
                )
            if mimi.get("failed_segments") != 0:
                invalid.append(f"{dataset} Mimi report contains failed segments")
            for field, expected in (
                ("codec_id", config.data.codec_id),
                ("codec_revision", config.data.codec_revision),
                ("mimi_weight_sha256", config.data.codec_weight_hash),
            ):
                if mimi.get(field) != expected:
                    invalid.append(f"{dataset} Mimi report {field} differs from config")
    if require_checkpoint:
        checkpoint = Path(require_checkpoint).expanduser()
        _required(checkpoint, "initial checkpoint", missing)
        if checkpoint.is_file():
            try:
                payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
                metadata = payload.get("metadata", {})
                if metadata.get("codec_id") != config.data.codec_id:
                    invalid.append("initial checkpoint codec_id differs from Canary config")
                if metadata.get("codec_revision") != config.data.codec_revision:
                    invalid.append("initial checkpoint codec_revision differs from Canary config")
                if metadata.get("codec_weight_hash") != config.data.codec_weight_hash:
                    invalid.append(
                        "initial checkpoint codec weight hash differs from Canary config"
                    )
                state = payload.get("model", {})
                if not isinstance(state, dict) or not state:
                    invalid.append("initial checkpoint has no model state")
            except (OSError, RuntimeError, ValueError, KeyError, TypeError) as error:
                invalid.append(f"initial checkpoint cannot be inspected: {error}")
    disk = shutil.disk_usage(root)
    result = {
        "dataset": dataset,
        "root": str(root),
        "passed": not missing and not invalid,
        "missing": missing,
        "invalid": invalid,
        "manifests": {split: str(path) for split, path in manifest_paths.items()},
        "shards": {split: [str(path) for path in paths] for split, paths in shard_paths.items()},
        "mimi_reports": mimi_reports,
        "free_disk_bytes": disk.free,
    }
    write_json(path("reports", "readiness.json"), result)
    if not result["passed"]:
        problems = "; ".join(missing + invalid)
        raise ValueError(f"training readiness failed: {problems}")
    return result
