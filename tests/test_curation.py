from __future__ import annotations

import json
from pathlib import Path

import pytest
from data import import_speech_manifest
from data.curation import (
    audit_canary_data,
    build_canary_manifest,
    build_canary_text,
    fetch_canary_data,
    select_canary_voices,
    synthesize_canary,
)
from data.curation.common import dataset_path, read_jsonl, registry_path, sha256_file
from data.curation.manifest import _duration_subset
from data.curation.prepare import prepare_canary_data
from data.curation.readiness import check_readiness
from model.types import SpeechMode
from runtime.config import DataConfig, ModelConfig, ProjectConfig, load_config


def _fixture_pipeline(root: Path) -> None:
    build_canary_text(root, fixture=True)
    synthesize_canary(root, fixture=True)
    build_canary_manifest(root, fixture=True)
    audit_canary_data(root, fixture=True)


def test_duration_subset_finds_a_quota_fit_that_greedy_order_misses() -> None:
    durations = [5.6, 5.6, 4.8, 4.8]
    selected = _duration_subset(durations, 10.4)
    assert sum(durations[index] for index in selected) == pytest.approx(10.4)


def test_fixture_pipeline_is_resumable(tmp_path: Path) -> None:
    root = tmp_path / "datasets"
    first = fetch_canary_data(root, fixture=True)
    select_canary_voices(root, fixture=True)
    _fixture_pipeline(root)
    manifest = dataset_path(root, "canary", "manifests", "episodes.jsonl")
    before = sha256_file(manifest)

    second = fetch_canary_data(root, fixture=True)
    _fixture_pipeline(root)

    assert first["inventory_sha256"] == second["inventory_sha256"]
    assert sha256_file(manifest) == before
    assert (dataset_path(root, "canary", "reports") / "data-card.md").is_file()
    assert (dataset_path(root, "canary", "reports") / "license-report.json").is_file()
    assert (dataset_path(root, "canary", "reports") / "quota-report.json").is_file()
    assert (dataset_path(root, "canary", "reports") / "quality-report.json").is_file()


def test_automatic_prepare_does_not_require_review_ledger(tmp_path: Path) -> None:
    root = tmp_path / "datasets"
    result = prepare_canary_data(root, config=ProjectConfig(), fixture=True)
    assert result["datasets"]["canary"]["audit"]["passed"]
    quality_path = dataset_path(root, "canary", "reports") / "quality-report.json"
    quality = json.loads(
        quality_path.read_text(encoding="utf-8")
    )
    assert quality["automatic_quality"]["mode"] == "automatic"
    assert not (root / "reviews" / "canary.jsonl").exists()


def test_canary_manifest_import_has_segment_aware_training_masks(
    tmp_path: Path,
) -> None:
    root = tmp_path / "datasets"
    fetch_canary_data(root, fixture=True)
    select_canary_voices(root, fixture=True)
    _fixture_pipeline(root)
    records = read_jsonl(dataset_path(root, "canary", "manifests", "episodes.jsonl"))
    public_record = next(record for record in records if record["category"] == "public_speech")
    dialogue_record = next(
        record for record in records if record["category"] == "synthetic_dialogue"
    )
    manifest = root / "single.jsonl"

    manifest.write_text(json.dumps(public_record) + "\n", encoding="utf-8")
    data = DataConfig()
    model = ModelConfig()
    public_episode = next(import_speech_manifest(manifest, data, model))
    assert not any(bool(unit.speech_codec_mask.item()) for unit in public_episode.units)
    assert all(unit.speech_mode.item() == SpeechMode.SILENCE for unit in public_episode.units)

    manifest.write_text(json.dumps(dialogue_record) + "\n", encoding="utf-8")
    dialogue = next(import_speech_manifest(manifest, data, model))
    modes = [unit.speech_mode.item() for unit in dialogue.units]
    masks = [bool(unit.speech_codec_mask.item()) for unit in dialogue.units]
    assert SpeechMode.SPEECH in modes
    assert SpeechMode.SILENCE in modes
    assert all(mask == (mode == SpeechMode.SPEECH) for mode, mask in zip(modes, masks, strict=True))


def test_manifest_interleaves_supervision_categories_deterministically(tmp_path: Path) -> None:
    root = tmp_path / "datasets"
    fetch_canary_data(root, fixture=True)
    select_canary_voices(root, fixture=True)
    build_canary_text(root, fixture=True)
    synthesize_canary(root, fixture=True)
    build_canary_manifest(root, fixture=True)

    train_path = dataset_path(root, "canary", "manifests", "train.jsonl")
    first = read_jsonl(train_path)
    first_hash = sha256_file(train_path)
    categories = [str(record["category"]) for record in first]
    # A short sequential run must not be trapped in public-speech-only data.
    assert len(set(categories[:8])) >= 2
    assert any(record.get("target_segments") for record in first[:8])
    counts = {category: categories.count(category) for category in set(categories)}

    build_canary_manifest(root, fixture=True)
    second = read_jsonl(train_path)
    assert sha256_file(train_path) == first_hash
    second_categories = [str(record["category"]) for record in second]
    assert {category: second_categories.count(category) for category in counts} == counts


def test_formal_fetch_requires_locked_source_lock(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="requires --lock"):
        fetch_canary_data(tmp_path / "datasets")
    assert registry_path(tmp_path / "datasets", "source-lock.template.json").is_file()


def test_formal_ppo_readiness_requires_sft_guard_assets(tmp_path: Path) -> None:
    config = load_config("configs/stages/canary-rl.yaml")
    with pytest.raises(ValueError, match="SFT replay"):
        check_readiness(tmp_path, config=config)


def test_readiness_reports_a_missing_data_root_cleanly(tmp_path: Path) -> None:
    config = load_config("configs/canary.yaml")
    missing = tmp_path / "absent-datasets"

    with pytest.raises(ValueError, match=f"data root: {missing}"):
        check_readiness(missing, config=config)


def test_canary_text_plan_meets_scale_and_duration_mix(tmp_path: Path) -> None:
    root = tmp_path / "datasets"
    report = build_canary_text(root)
    plans = json.loads(Path(report["path"]).read_text(encoding="utf-8"))["plans"]

    assert len(plans) == 120
    assert sum(plan["language"] == "zh" for plan in plans) == 96
    assert sum(plan["language"] == "en" for plan in plans) == 24
    assert sum(plan["duration_class"] == "short" for plan in plans) == 72
    assert sum(plan["duration_class"] == "medium" for plan in plans) == 30
    assert sum(plan["duration_class"] == "long" for plan in plans) == 18
    assert report["quality_status"] == "generated"
