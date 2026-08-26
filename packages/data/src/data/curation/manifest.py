from __future__ import annotations

import json
import shlex
import subprocess
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from data.curation.audio import (
    FRAME_SAMPLES,
    SAMPLE_RATE,
    align_up,
    read_mono,
    write_flac,
)
from data.curation.common import (
    CATEGORIES,
    LANGUAGES,
    SPLITS,
    dataset_path,
    ensure_tree,
    read_json,
    read_jsonl,
    registry_path,
    relative_to_root,
    sha256_file,
    stable_hash,
    write_json,
    write_jsonl,
)
from data.curation.spec import (
    CATEGORY_FRACTIONS,
    LANGUAGE_FRACTIONS,
    SPLIT_FRACTIONS,
    dataset_spec,
)

# Episode audio is quantized to PCM16 by ``write_flac``.  Keep a small amount
# of headroom so quantization cannot turn a nominally safe source peak into a
# value above the audit gate's -1 dBFS limit.
AUDIO_PEAK_LIMIT_DBFS = -1.5
AUDIO_PEAK_LIMIT = 10 ** (AUDIO_PEAK_LIMIT_DBFS / 20.0)


def build_source_inventory(command: str, root: str | Path) -> dict[str, Any]:
    """Build the normalized public-source inventory before synthesis starts."""
    root = Path(root).expanduser().resolve()
    ensure_tree(root)
    output = registry_path(root, "normalized", "source-items.jsonl")
    receipt_path = registry_path(root, "normalized", "source-items.receipt.json")
    fetch_report = registry_path(root, "reports", "fetch-report.json")
    adapter_files = {}
    for argument in shlex.split(command):
        candidate = Path(argument).expanduser()
        if candidate.is_file():
            resolved = candidate.resolve()
            adapter_files[str(resolved)] = sha256_file(resolved)
    recipe = {
        "operation": "index-and-normalize",
        "fetch_report_sha256": sha256_file(fetch_report),
        "sample_rate": SAMPLE_RATE,
        "channels": 1,
        "format": "FLAC/PCM16",
        "adapter": command,
        "adapter_files": adapter_files,
    }
    recipe_hash = stable_hash(recipe)
    if output.is_file() and receipt_path.is_file():
        receipt = read_json(receipt_path)
        if receipt.get("recipe_sha256") == recipe_hash and receipt.get(
            "inventory_sha256"
        ) == sha256_file(output):
            return {**receipt, "path": str(output), "cached": True}
    with tempfile.NamedTemporaryFile("w", suffix=".json", encoding="utf-8") as request_file:
        json.dump(
            {
                **recipe,
                "raw_root": str(root.parent / "assets" / "sources"),
                "licenses_root": str(registry_path(root, "licenses")),
                "data_root": str(root),
                "output": str(output),
            },
            request_file,
        )
        request_file.flush()
        completed = subprocess.run(
            [*shlex.split(command), "--request", request_file.name, "--output", str(output)],
            capture_output=True,
            text=True,
            check=False,
        )
    if completed.returncode:
        raise RuntimeError(
            f"normalization adapter failed: {(completed.stderr or completed.stdout)[-1000:]}"
        )
    if not output.is_file():
        raise RuntimeError("normalization adapter did not create source-items.jsonl")
    write_json(
        receipt_path,
        {"recipe_sha256": recipe_hash, "inventory_sha256": sha256_file(output)},
    )
    return {
        "path": str(output),
        "inventory_sha256": sha256_file(output),
        "recipe_sha256": recipe_hash,
        "cached": False,
    }


def _load_assets(
    root: Path, dataset: str
) -> tuple[
    list[dict[str, Any]], dict[str, Any], dict[tuple[str, str], dict[str, Any]], dict[str, Any]
]:
    source_path = registry_path(root, "normalized", "source-items.jsonl")
    plans_path = dataset_path(root, dataset, "text", "plans.json")
    synthesis_path = dataset_path(root, dataset, "synthesized", "utterances.jsonl")
    voices_path = registry_path(root, "voices", "registry.json")
    for path in (source_path, plans_path, synthesis_path, voices_path):
        if not path.is_file():
            raise FileNotFoundError(f"required Canary asset is absent: {path}")
    source_items = read_jsonl(source_path)
    for item in source_items:
        audio = Path(item["audio"])
        if not audio.is_absolute():
            audio = root / audio
        if not audio.is_file() or sha256_file(audio) != item.get("audio_sha256"):
            raise ValueError(f"source inventory audio hash mismatch: {item.get('source_item_id')}")
    receipt_rows = read_jsonl(synthesis_path)
    for item in receipt_rows:
        audio = Path(item["audio"])
        if not audio.is_absolute():
            audio = root / audio
        if not audio.is_file() or sha256_file(audio) != item.get("audio_sha256"):
            raise ValueError(
                "synthesis receipt audio hash mismatch: "
                f"{item.get('plan_id') or item.get('source_item_id')}/{item.get('turn_id')}"
            )
    receipts = {
        (str(item.get("plan_id") or item.get("source_item_id")), str(item["turn_id"])): item
        for item in receipt_rows
    }
    if len(receipts) != len(receipt_rows):
        raise ValueError("synthesis receipt keys are duplicated")
    registry = read_json(voices_path)
    registry_hash = registry.get("registry_sha256")
    expected_registry = stable_hash(
        {key: value for key, value in registry.items() if key != "registry_sha256"}
    )
    if registry_hash != expected_registry:
        raise ValueError("voice registry hash is stale")
    return source_items, read_json(plans_path), receipts, registry


def _cached_episode(path: Path, recipe_hash: str) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    record = read_json(path)
    if (
        record.get("recipe_sha256") != recipe_hash
    ):
        return None
    for field in ("mic_audio", "target_speech"):
        asset = Path(record[field])
        if not asset.is_file() or sha256_file(asset) != record.get(f"{field}_sha256"):
            return None
    if record.get("screens"):
        screen = Path(record["screens"])
        if not screen.is_file() or sha256_file(screen) != record.get("screens_sha256"):
            return None
    return record


def _write_episode_audio(
    mic_path: Path, target_path: Path, mic: np.ndarray, target: np.ndarray
) -> None:
    """Write a timeline with one common peak scale for mic and target."""
    peak = max(
        float(np.abs(mic).max(initial=0.0)),
        float(np.abs(target).max(initial=0.0)),
    )
    if peak > AUDIO_PEAK_LIMIT:
        scale = AUDIO_PEAK_LIMIT / peak
        mic = mic * scale
        target = target * scale
    write_flac(mic_path, mic)
    write_flac(target_path, target)


def _record_path(root: Path, dataset: str, episode_id: str) -> Path:
    return dataset_path(root, dataset, "normalized", "episodes", f"{episode_id}.json")


def _minimum_episode_ticks(record: dict[str, Any]) -> int:
    """Return the shortest timeline that preserves speech and STOP labels."""
    ends = [int(turn["end_sample"]) for turn in record.get("turns", [])]
    minimum = max(1, max((align_up(end) // FRAME_SAMPLES for end in ends), default=1))
    for segment in record.get("target_segments", []):
        # Audit requires a complete frame after the frame containing speech.
        stop_frame = (int(segment["end_sample"]) - 1) // FRAME_SAMPLES + 1
        minimum = max(minimum, stop_frame + 1)
    return minimum


def _calibrate_split_durations(root: Path, dataset: str, records: list[dict[str, Any]]) -> None:
    """Trim only deterministic tail silence to hit each split's tick quota."""
    spec = dataset_spec(dataset)
    tick_seconds = FRAME_SAMPLES / SAMPLE_RATE
    durations: dict[str, int] = {}
    capacities: dict[str, int] = {}
    buckets: dict[str, tuple[str, str, str]] = {}
    cross_ticks: dict[tuple[str, str, str], int] = defaultdict(int)
    category_ticks: dict[str, int] = defaultdict(int)
    language_ticks: dict[str, int] = defaultdict(int)
    for record in records:
        episode_id = str(record["episode_id"])
        ticks = read_mono(Path(record["mic_audio"])).size // FRAME_SAMPLES
        key = (str(record["category"]), str(record["language"]), str(record["split"]))
        durations[episode_id] = ticks
        capacities[episode_id] = max(0, ticks - _minimum_episode_ticks(record))
        buckets[episode_id] = key
        cross_ticks[key] += ticks
        category_ticks[key[0]] += ticks
        language_ticks[key[1]] += ticks

    def error(value: int, target: float) -> float:
        return ((value - target) / max(target, 1.0)) ** 2

    def trim_cost(key: tuple[str, str, str]) -> float:
        category, language, split = key
        cross_target = spec.target_seconds(category, language, split) / tick_seconds
        category_target = spec.duration_seconds * CATEGORY_FRACTIONS[category] / tick_seconds
        language_target = spec.duration_seconds * LANGUAGE_FRACTIONS[language] / tick_seconds
        return (
            error(cross_ticks[key] - 1, cross_target)
            - error(cross_ticks[key], cross_target)
            + error(category_ticks[category] - 1, category_target)
            - error(category_ticks[category], category_target)
            + error(language_ticks[language] - 1, language_target)
            - error(language_ticks[language], language_target)
        )

    for split in SPLITS:
        split_records = sorted(
            (record for record in records if record["split"] == split),
            key=lambda record: str(record["episode_id"]),
        )
        if not split_records:
            continue
        current_ticks = sum(durations[str(record["episode_id"])] for record in split_records)
        target_ticks = round(spec.duration_seconds * SPLIT_FRACTIONS[split] / tick_seconds)
        excess = current_ticks - target_ticks
        if excess <= 0:
            continue
        bucket_capacity: dict[tuple[str, str, str], int] = defaultdict(int)
        for record in split_records:
            episode_id = str(record["episode_id"])
            bucket_capacity[buckets[episode_id]] += capacities[episode_id]
        requested: dict[tuple[str, str, str], int] = defaultdict(int)
        for _ in range(excess):
            candidates = [key for key, capacity in bucket_capacity.items() if capacity > 0]
            if not candidates:
                raise ValueError(
                    f"cannot calibrate {dataset}/{split} duration: no tail silence remains"
                )
            key = min(candidates, key=lambda candidate: (trim_cost(candidate), candidate))
            requested[key] += 1
            bucket_capacity[key] -= 1
            cross_ticks[key] -= 1
            category_ticks[key[0]] -= 1
            language_ticks[key[1]] -= 1

        for key, requested_ticks in sorted(requested.items()):
            candidates = sorted(
                (
                    (capacities[str(record["episode_id"])], str(record["episode_id"]), record)
                    for record in split_records
                    if buckets[str(record["episode_id"])] == key
                    and capacities[str(record["episode_id"])] > 0
                ),
                key=lambda item: (-item[0], item[1]),
            )
            for capacity, _, record in candidates:
                if requested_ticks <= 0:
                    break
                trim = min(requested_ticks, capacity)
                mic_path = Path(record["mic_audio"])
                target_path = Path(record["target_speech"])
                mic = read_mono(mic_path)[: -trim * FRAME_SAMPLES]
                target = read_mono(target_path)[: -trim * FRAME_SAMPLES]
                _write_episode_audio(mic_path, target_path, mic, target)
                record["mic_audio_sha256"] = sha256_file(mic_path)
                record["target_speech_sha256"] = sha256_file(target_path)
                record["timeline_trimmed_ticks"] = trim
                write_json(_record_path(root, dataset, record["episode_id"]), record)
                requested_ticks -= trim
            if requested_ticks:
                raise ValueError(
                    f"cannot calibrate {dataset}/{split}/{key}: {requested_ticks} ticks remain"
                )


def _screens(
    root: Path,
    dataset: str,
    episode_id: str,
    ticks: int,
    fixture: bool,
    screen_command: str | None,
) -> str:
    output = dataset_path(root, dataset, "normalized", "screens", f"{episode_id}.npz")
    receipt_path = output.with_suffix(".json")
    output.parent.mkdir(parents=True, exist_ok=True)
    recipe = {
        "episode_id": episode_id,
        "ticks": ticks,
        "tick_ms": 80,
        "height": 224,
        "width": 224,
        "fixture": fixture,
        "adapter": screen_command,
    }
    recipe_hash = stable_hash(recipe)
    if output.exists() and receipt_path.exists():
        receipt = read_json(receipt_path)
        if receipt.get("recipe_sha256") == recipe_hash and receipt.get(
            "artifact_sha256"
        ) == sha256_file(output):
            return relative_to_root(output, root)
    if fixture:
        frame = np.zeros((1, 3, 224, 224), dtype=np.float32)
        frame[:, 0, 48:176, 48:176] = 0.25
        np.savez_compressed(output, ticks=np.asarray([min(1, ticks - 1)]), frames=frame)
    else:
        if not screen_command:
            raise ValueError("screen-conditioned episodes require --screen-command")
        request = {"operation": "capture-isolated-sandbox", **recipe, "output": str(output)}
        with tempfile.NamedTemporaryFile("w", suffix=".json", encoding="utf-8") as handle:
            json.dump(request, handle)
            handle.flush()
            result = subprocess.run(
                [
                    *shlex.split(screen_command),
                    "--request",
                    handle.name,
                    "--output",
                    str(output),
                ],
                capture_output=True,
                text=True,
                check=False,
            )
        if result.returncode or not output.is_file():
            raise RuntimeError(f"screen adapter failed: {(result.stderr or result.stdout)[-1000:]}")
    write_json(
        receipt_path,
        {"recipe_sha256": recipe_hash, "artifact_sha256": sha256_file(output)},
    )
    return relative_to_root(output, root)


def _compose_plan(
    root: Path,
    dataset: str,
    plan: dict[str, Any],
    receipts: dict[tuple[str, str], dict[str, Any]],
    registry: dict[str, Any],
    fixture: bool,
    screen_command: str | None,
) -> dict[str, Any]:
    episode_id = f"{dataset}-plan-{plan['plan_id']}"
    turn_receipts = []
    for turn in plan["turns"]:
        key = (str(plan["plan_id"]), str(turn["turn_id"]))
        if key not in receipts:
            raise ValueError(f"synthesis receipt is absent for {key}")
        turn_receipts.append(receipts[key])
    recipe_hash = stable_hash(
        {
            "plan": plan["recipe_sha256"],
            "receipts": [
                {
                    "recipe_sha256": receipt["recipe_sha256"],
                    "audio_sha256": receipt["audio_sha256"],
                }
                for receipt in turn_receipts
            ],
            "voice_registry": registry["registry_sha256"],
            "screen_adapter": screen_command if plan["category"] == "screen_task" else None,
            "fixture": fixture,
        }
    )
    audio_dir = dataset_path(root, dataset, "normalized", "episodes")
    record_path = audio_dir / f"{episode_id}.json"
    cached = _cached_episode(record_path, recipe_hash)
    if cached is not None:
        return cached
    cursor = 2 * FRAME_SAMPLES
    mic = np.zeros(
        align_up(int(float(plan["target_duration_seconds"]) * SAMPLE_RATE)), dtype=np.float32
    )
    target = np.zeros_like(mic)
    turns = []
    segments = []
    user_voice_id = ""
    for turn in plan["turns"]:
        key = (str(plan["plan_id"]), str(turn["turn_id"]))
        receipt = receipts[key]
        waveform = read_mono(root / receipt["audio"])
        if turn["role"] == "assistant":
            cursor = align_up(cursor + 3 * FRAME_SAMPLES)
            end = cursor + waveform.size
            needed = align_up(end) + FRAME_SAMPLES
            if needed > target.size:
                extension = needed - target.size
                mic = np.pad(mic, (0, extension))
                target = np.pad(target, (0, extension))
            target[cursor:end] = waveform
            segments.append({"turn_id": turn["turn_id"], "start_sample": cursor, "end_sample": end})
        else:
            cursor = align_up(cursor)
            end = cursor + waveform.size
            if end > mic.size:
                extension = align_up(end) + FRAME_SAMPLES - mic.size
                mic = np.pad(mic, (0, extension))
                target = np.pad(target, (0, extension))
            mic[cursor:end] = waveform
            user_voice_id = str(receipt["voice_id"])
        turns.append(
            {
                "turn_id": turn["turn_id"],
                "role": turn["role"],
                "text": turn["text"],
                "start_sample": cursor,
                "end_sample": end,
            }
        )
        cursor = end
    timeline = max(mic.size, align_up(cursor) + 3 * FRAME_SAMPLES)
    mic = np.pad(mic, (0, timeline - mic.size))
    target = np.pad(target, (0, timeline - target.size))
    mic_path = audio_dir / f"{episode_id}-mic.flac"
    target_path = audio_dir / f"{episode_id}-target.flac"
    _write_episode_audio(mic_path, target_path, mic, target)
    source_license = "internal-generated-plans; CosyVoice example voices"
    license_hash = sha256_file(registry_path(root, "voices", "registry.json"))
    record = {
        "episode_id": episode_id,
        "plan_id": plan["plan_id"],
        "mic_audio": str(mic_path.resolve()),
        "mic_audio_sha256": sha256_file(mic_path),
        "target_speech": str(target_path.resolve()),
        "target_speech_sha256": sha256_file(target_path),
        "source": "generated-computer-dialogue",
        "source_version": "canary-plan-v1",
        "source_url": "internal://datasets/text-plans",
        "source_utterance_ids": [plan["plan_id"]],
        "source_license": source_license,
        "redistribution_allowed": False,
        "license_sha256": license_hash,
        "template_id": plan["template_id"],
        "intent": plan["intent"],
        "language": plan["language"],
        "split": plan["split"],
        "user_voice_id": user_voice_id,
        "assistant_voice_id": registry["assistant_voice_id"],
        "session_id_hash": stable_hash({"scenario": plan["scenario_id"]}),
        "device_id_hash": "isolated-sandbox" if plan["category"] == "screen_task" else "synthetic",
        "scenario": plan["scenario_id"],
        "category": plan["category"],
        "turns": turns,
        "target_segments": segments,
        "recipe_sha256": recipe_hash,
        "fixture": fixture,
    }
    if plan["category"] == "screen_task":
        screen_path = _screens(
            root, dataset, episode_id, timeline // FRAME_SAMPLES, fixture, screen_command
        )
        record["screens"] = str((root / screen_path).resolve())
        record["screens_sha256"] = sha256_file(root / screen_path)
    write_json(record_path, record)
    return record


def _compose_source(
    root: Path,
    dataset: str,
    item: dict[str, Any],
    receipts: dict[tuple[str, str], dict[str, Any]],
    registry: dict[str, Any],
    fixture: bool,
) -> dict[str, Any]:
    episode_id = f"{dataset}-source-{item['source_item_id']}"
    response = receipts.get((str(item["source_item_id"]), "assistant-response"))
    recipe_hash = stable_hash(
        {
            "source_audio": item["audio_sha256"],
            "source_item": item["source_item_id"],
            "response": response.get("recipe_sha256") if response else None,
            "response_audio": response.get("audio_sha256") if response else None,
            "voice_registry": registry["registry_sha256"],
            "dataset": dataset,
            "fixture": fixture,
        }
    )
    audio_dir = dataset_path(root, dataset, "normalized", "episodes")
    record_path = audio_dir / f"{episode_id}.json"
    cached = _cached_episode(record_path, recipe_hash)
    if cached is not None:
        return cached
    source_audio = read_mono(root / item["audio"])
    lead = 2 * FRAME_SAMPLES
    mic_size = align_up(lead + source_audio.size) + 4 * FRAME_SAMPLES
    mic = np.zeros(mic_size, dtype=np.float32)
    target = np.zeros(mic_size, dtype=np.float32)
    mic[lead : lead + source_audio.size] = source_audio
    turns = [
        {
            "turn_id": "source-input",
            "role": "user",
            "text": item.get("text", ""),
            "start_sample": lead,
            "end_sample": lead + source_audio.size,
        }
    ]
    segments: list[dict[str, Any]] = []
    if item["category"] == "adjacent_turns":
        receipt = response
        if receipt is None:
            raise ValueError(f"adjacent response receipt is absent for {item['source_item_id']}")
        waveform = read_mono(root / receipt["audio"])
        start = align_up(lead + source_audio.size + 4 * FRAME_SAMPLES)
        end = start + waveform.size
        timeline = align_up(end) + 4 * FRAME_SAMPLES
        mic = np.pad(mic, (0, max(0, timeline - mic.size)))
        target = np.pad(target, (0, max(0, timeline - target.size)))
        target[start:end] = waveform
        turns.append(
            {
                "turn_id": "assistant-response",
                "role": "assistant",
                "text": item["response_text"],
                "start_sample": start,
                "end_sample": end,
            }
        )
        segments.append({"turn_id": "assistant-response", "start_sample": start, "end_sample": end})
    mic_path = audio_dir / f"{episode_id}-mic.flac"
    target_path = audio_dir / f"{episode_id}-target.flac"
    _write_episode_audio(mic_path, target_path, mic, target)
    record = {
        "episode_id": episode_id,
        "mic_audio": str(mic_path.resolve()),
        "mic_audio_sha256": sha256_file(mic_path),
        "target_speech": str(target_path.resolve()),
        "target_speech_sha256": sha256_file(target_path),
        "source": item["source_id"],
        "source_version": item["source_version"],
        "source_url": item["source_url"],
        "source_utterance_ids": item["source_utterance_ids"],
        "source_license": item["source_license"],
        "redistribution_allowed": item["redistribution_allowed"],
        "license_sha256": item["license_sha256"],
        "template_id": (
            f"source-{'adjacent' if segments else 'input'}-"
            f"{item['source_id']}-{item['language']}-{item['split']}-v1"
        ),
        "intent": "conversational-response" if segments else "listen-observe",
        "language": item["language"],
        "split": item["split"],
        "user_voice_id": item.get("speaker_id", "unknown-source-speaker"),
        "assistant_voice_id": registry["assistant_voice_id"],
        "session_id_hash": stable_hash(
            {"source": item["source_id"], "session": item["session_id"]}
        ),
        "device_id_hash": str(item["source_id"]),
        "scenario": str(item["source_item_id"]),
        "category": item["category"],
        "turns": turns,
        "target_segments": segments,
        "recipe_sha256": recipe_hash,
        "source_normalization": item.get("normalization"),
        "fixture": fixture,
    }
    write_json(record_path, record)
    return record


def _verify_isolation(records: list[dict[str, Any]]) -> None:
    dimensions = {
        "speaker": "user_voice_id",
        "session": "session_id_hash",
        "template": "template_id",
        "scenario": "scenario",
    }
    for label, field in dimensions.items():
        seen: dict[str, set[str]] = defaultdict(set)
        for record in records:
            seen[str(record[field])].add(str(record["split"]))
        leaks = [value for value, splits in seen.items() if len(splits) > 1]
        if leaks:
            raise ValueError(f"{label} values cross dataset splits: {leaks[:3]}")


def _duration_seconds(root: Path, record: dict[str, Any]) -> float:
    path = Path(record["mic_audio"])
    if not path.is_absolute():
        path = root / path
    return read_mono(path).size / SAMPLE_RATE


def _duration_subset(durations: list[float], target: float) -> list[int]:
    tick_seconds = FRAME_SAMPLES / SAMPLE_RATE
    ticks = [round(duration / tick_seconds) for duration in durations]
    target_ticks = target / tick_seconds
    minimum = int(np.ceil(target_ticks * 0.98))
    maximum = int(np.floor(target_ticks * 1.02))
    reachable = bytearray(maximum + 1)
    previous = [-1] * (maximum + 1)
    chosen = [-1] * (maximum + 1)
    reachable[0] = 1
    for index, duration_ticks in enumerate(ticks):
        if duration_ticks <= 0 or duration_ticks > maximum:
            continue
        for elapsed in range(maximum - duration_ticks, -1, -1):
            updated = elapsed + duration_ticks
            if reachable[elapsed] and not reachable[updated]:
                reachable[updated] = 1
                previous[updated] = elapsed
                chosen[updated] = index
    candidates = [elapsed for elapsed in range(minimum, maximum + 1) if reachable[elapsed]]
    if not candidates:
        return []
    elapsed = min(candidates, key=lambda value: (abs(value - target_ticks), value))
    indices = []
    while elapsed:
        index = chosen[elapsed]
        if index < 0:
            raise RuntimeError("duration subset reconstruction failed")
        indices.append(index)
        elapsed = previous[elapsed]
    return sorted(indices)


def _select_sources(
    root: Path,
    dataset: str,
    source_items: list[dict[str, Any]],
    receipts: dict[tuple[str, str], dict[str, Any]],
    registry: dict[str, Any],
    excluded_sources: set[str],
    fixture: bool,
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for item in source_items:
        source_ids = set(map(str, item.get("source_utterance_ids", [])))
        if source_ids & excluded_sources:
            continue
        grouped[(item["category"], item["language"], item["split"])].append(item)
    spec = dataset_spec(dataset)
    for key, items in grouped.items():
        items.sort(key=lambda value: value["source_item_id"])
        if fixture:
            candidates = items[::2] if dataset == "canary" else items
        else:
            candidates = [
                item
                for item in items
                if item["category"] != "adjacent_turns"
                or (str(item["source_item_id"]), "assistant-response") in receipts
            ]
        target = spec.target_seconds(*key)
        records = [
            _compose_source(root, dataset, item, receipts, registry, fixture) for item in candidates
        ]
        if fixture:
            selected.extend(records)
            continue
        durations = [_duration_seconds(root, record) for record in records]
        indices = _duration_subset(durations, target)
        if not indices:
            raise ValueError(
                f"not enough independent source audio for {key}: no combination within "
                f"{target * 0.98:.1f}-{target * 1.02:.1f}s"
            )
        selected.extend(records[index] for index in indices)
    return selected


_TRAINING_CATEGORY_ORDER = (
    "synthetic_dialogue",
    "adjacent_turns",
    "screen_task",
    "public_speech",
)


def _interleave_categories(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Interleave categories while preserving each category's input order.

    The training reader is intentionally sequential so episode state can carry
    across TBPTT chunks.  A source-grouped manifest can therefore hide all
    assistant supervision behind a long public-speech prefix.  Deterministic
    round-robin ordering exposes every available category early without
    duplicating, splitting, or otherwise changing any episode.
    """
    grouped: dict[str, list[dict[str, Any]]] = {category: [] for category in CATEGORIES}
    unknown: list[dict[str, Any]] = []
    for record in records:
        category = str(record.get("category", ""))
        if category in grouped:
            grouped[category].append(record)
        else:
            unknown.append(record)

    # Keep an explicit order for the known formal categories.  Unknown
    # categories are retained after the round-robin sequence rather than being
    # silently dropped from a manifest.
    ordered_categories = [category for category in _TRAINING_CATEGORY_ORDER if grouped[category]]
    ordered: list[dict[str, Any]] = []
    index = 0
    while ordered_categories:
        next_categories: list[str] = []
        for category in ordered_categories:
            bucket = grouped[category]
            if index < len(bucket):
                ordered.append(bucket[index])
            if index + 1 < len(bucket):
                next_categories.append(category)
        ordered_categories = next_categories
        index += 1
    ordered.extend(unknown)
    return ordered


def build_canary_manifest(
    root: str | Path,
    *,
    fixture: bool = False,
    normalize_command: str | None = None,
    screen_command: str | None = None,
) -> dict[str, Any]:
    dataset = "canary"
    root = Path(root).expanduser().resolve()
    ensure_tree(root)
    if normalize_command:
        build_source_inventory(normalize_command, root)
    elif not fixture:
        inventory = registry_path(root, "normalized", "source-items.jsonl")
        receipt = registry_path(root, "normalized", "source-items.receipt.json")
        if not inventory.is_file() or not receipt.is_file():
            raise ValueError(
                "formal Canary manifest construction requires --normalize-command on first run"
            )
    source_items, plans, receipts, registry = _load_assets(root, dataset)
    records = _select_sources(
        root,
        dataset,
        source_items,
        receipts,
        registry,
        set(),
        fixture,
    )
    for plan in plans["plans"]:
        records.append(
            _compose_plan(root, dataset, plan, receipts, registry, fixture, screen_command)
        )
    ids = [record["episode_id"] for record in records]
    if len(ids) != len(set(ids)):
        raise ValueError("episode IDs are duplicated")
    _calibrate_split_durations(root, dataset, records)
    _verify_isolation(records)
    destination = dataset_path(root, dataset, "manifests")
    # Keep the global and split manifests in the same deterministic training
    # order.  The order is part of the streaming data contract: short runs
    # must observe speech-positive and silent episodes alike.
    records = [
        record
        for split in SPLITS
        for record in _interleave_categories(
            [candidate for candidate in records if candidate["split"] == split]
        )
    ]
    for split in SPLITS:
        split_records = [r for r in records if r["split"] == split]
        write_jsonl(destination / f"{split}.jsonl", split_records)
    manifest_path = destination / "episodes.jsonl"
    write_jsonl(manifest_path, records)
    quota = defaultdict(float)
    for record in records:
        samples = read_mono(Path(record["mic_audio"])).size
        quota[(record["category"], record["language"], record["split"])] += samples / SAMPLE_RATE
    report = {
        "dataset": dataset,
        "fixture": fixture,
        "episodes": len(records),
        "manifest": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "quotas": [
            {
                "category": category,
                "language": language,
                "split": split,
                "actual_seconds": quota[(category, language, split)],
                "target_seconds": dataset_spec(dataset).target_seconds(category, language, split),
            }
            for category in CATEGORIES
            for language in LANGUAGES
            for split in SPLITS
        ],
    }
    write_json(dataset_path(root, dataset, "reports", "manifest.json"), report)
    return report
