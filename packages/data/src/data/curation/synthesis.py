from __future__ import annotations

import json
import shlex
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from data.curation.audio import fixture_voice, quality_metrics, write_flac
from data.curation.common import (
    dataset_path,
    ensure_tree,
    read_json,
    registry_path,
    relative_to_root,
    require_sha256,
    sha256_file,
    stable_hash,
    write_json,
    write_jsonl,
)
from data.curation.text import plan_recipe_sha256


def _run_adapter(command: str, request: dict[str, Any], output: Path) -> None:
    arguments = shlex.split(command)
    if not arguments:
        raise ValueError("adapter command is empty")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", suffix=".json", encoding="utf-8") as request_file:
        json.dump(request, request_file, ensure_ascii=False)
        request_file.flush()
        completed = subprocess.run(
            [*arguments, "--request", request_file.name, "--output", str(output)],
            check=False,
            capture_output=True,
            text=True,
        )
    if completed.returncode:
        detail = (completed.stderr or completed.stdout).strip()
        raise RuntimeError(f"adapter failed ({completed.returncode}): {detail[-1000:]}")
    if not output.is_file():
        raise RuntimeError("adapter completed without creating its requested output")


def _asr_score(
    command: str | None, path: Path, text: str, language: str, fixture: bool
) -> tuple[str, float]:
    metric = "cer" if language == "zh" else "wer"
    if fixture:
        return metric, 0.0
    if not command:
        raise ValueError("formal Canary synthesis requires --asr-command for CER/WER gating")
    with tempfile.TemporaryDirectory() as temporary:
        result_path = Path(temporary) / "asr.json"
        _run_adapter(
            command,
            {"operation": "transcribe", "audio": str(path), "text": text, "language": language},
            result_path,
        )
        result = read_json(result_path)
    if result.get("metric") != metric:
        raise ValueError(f"ASR adapter must return the {metric} metric for {language}")
    score = float(result["score"])
    if not 0 <= score <= 1:
        raise ValueError("ASR score must be in [0, 1]")
    return metric, score


def _formal_audio_report(path: Path) -> dict[str, Any]:
    report_path = path.with_suffix(".metrics.json")
    if not report_path.is_file():
        raise ValueError(
            "formal Canary TTS adapter must write a sibling .metrics.json with integrated_lufs"
        )
    report = read_json(report_path)
    loudness = float(report["integrated_lufs"])
    if not -24.0 <= loudness <= -22.0:
        raise ValueError(f"synthesized loudness {loudness:.2f} LUFS is outside -23 +/- 1")
    return {"integrated_lufs": loudness, "metrics_sha256": sha256_file(report_path)}


def _cached_synthesis(
    root: Path, recipe: dict[str, Any], output: Path
) -> tuple[dict[str, Any], dict[str, Any]] | None:
    content_recipe = {
        key: recipe[key]
        for key in (
            "text",
            "language",
            "role",
            "voice_id",
            "voice_prompt_sha256",
            "model_sha256",
        )
    }
    cache_key = stable_hash(content_recipe)
    cache_audio = registry_path(root, "synthesis-cache", f"{cache_key}.flac")
    cache_receipt = cache_audio.with_suffix(".json")
    if not cache_audio.is_file() or not cache_receipt.is_file():
        return None
    receipt = read_json(cache_receipt)
    if (
        receipt.get("content_recipe_sha256") != cache_key
        or receipt.get("audio_sha256") != sha256_file(cache_audio)
    ):
        return None
    output.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(cache_audio, output)
    shutil.copy2(cache_audio.with_suffix(".metrics.json"), output.with_suffix(".metrics.json"))
    return receipt, content_recipe


def _store_synthesis_cache(
    root: Path,
    recipe: dict[str, Any],
    output: Path,
    *,
    metric: str,
    score: float,
    attempts: int,
) -> None:
    content_recipe = {
        key: recipe[key]
        for key in (
            "text",
            "language",
            "role",
            "voice_id",
            "voice_prompt_sha256",
            "model_sha256",
        )
    }
    cache_key = stable_hash(content_recipe)
    cache_audio = registry_path(root, "synthesis-cache", f"{cache_key}.flac")
    cache_audio.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(output, cache_audio)
    shutil.copy2(output.with_suffix(".metrics.json"), cache_audio.with_suffix(".metrics.json"))
    write_json(
        cache_audio.with_suffix(".json"),
        {
            "content_recipe_sha256": cache_key,
            "audio_sha256": sha256_file(cache_audio),
            "asr_metric": metric,
            "asr_score": score,
            "attempts": attempts,
        },
    )


def synthesize_canary(
    root: str | Path,
    *,
    fixture: bool = False,
    synth_command: str | None = None,
    asr_command: str | None = None,
    model_sha256: str | None = None,
) -> dict[str, Any]:
    dataset = "canary"
    root = Path(root).expanduser().resolve()
    ensure_tree(root)
    plans_path = dataset_path(root, dataset, "text", "plans.json")
    voice_registry_path = registry_path(root, "voices", "registry.json")
    plans = read_json(plans_path)
    registry = read_json(voice_registry_path)
    if plans.get("dataset") != dataset:
        raise ValueError("text plan dataset does not match the requested dataset")
    stale = [
        plan["plan_id"]
        for plan in plans["plans"]
        if plan.get("recipe_sha256") != plan_recipe_sha256(plan)
    ]
    if stale:
        raise ValueError(f"text plan recipe hashes are stale: {stale[:3]}")
    if fixture:
        model_hash = "fixture"
    else:
        model_hash = require_sha256(model_sha256, "TTS model")
        if not synth_command:
            raise ValueError("formal Canary synthesis requires --synth-command")
    voices = {voice["voice_id"]: voice for voice in registry["voices"]}
    assistant_voice = str(registry["assistant_voice_id"])
    user_voices = {
        (language, split): sorted(
            voice["voice_id"]
            for voice in voices.values()
            if voice.get("role") == "user"
            and voice.get("language") in {language, "multilingual"}
            and voice.get("split") == split
        )
        for language in ("zh", "en")
        for split in ("train", "validation", "test")
    }
    receipts: list[dict[str, Any]] = []
    rejected = 0
    for plan_index, plan in enumerate(plans["plans"]):
        language = str(plan["language"])
        split = str(plan["split"])
        available_voices = user_voices[(language, split)]
        if not available_voices:
            raise ValueError(f"no {language} user voice is assigned to split {split}")
        user_voice = available_voices[plan_index % len(available_voices)]
        for turn in plan["turns"]:
            role = str(turn["role"])
            voice_id = assistant_voice if role == "assistant" else user_voice
            prompt_hash = str(voices[voice_id]["prompt_sha256"])
            prompt_audio = root / str(voices[voice_id]["prompt_audio"])
            recipe = {
                "plan_id": plan["plan_id"],
                "plan_recipe_sha256": plan["recipe_sha256"],
                "turn_id": turn["turn_id"],
                "text": turn["text"],
                "language": language,
                "role": role,
                "voice_id": voice_id,
                "voice_prompt_sha256": prompt_hash,
                "voice_prompt_audio": str(prompt_audio.resolve()),
                "voice_prompt_text": str(voices[voice_id].get("prompt_text") or ""),
                "model_sha256": model_hash,
            }
            recipe_hash = stable_hash(recipe)
            output = dataset_path(
                root, dataset, "synthesized", plan["plan_id"], f"{turn['turn_id']}.flac"
            )
            receipt_path = output.with_suffix(".json")
            if output.is_file() and receipt_path.is_file():
                old = read_json(receipt_path)
                if old.get("recipe_sha256") == recipe_hash and old.get(
                    "audio_sha256"
                ) == sha256_file(output):
                    receipts.append(old)
                    continue
            cached = None if fixture else _cached_synthesis(root, recipe, output)
            if cached:
                cache_receipt, _ = cached
                attempts = int(cache_receipt["attempts"])
                metric = str(cache_receipt["asr_metric"])
                score = float(cache_receipt["asr_score"])
                metrics = quality_metrics(output)
            else:
                attempts = 0
                while True:
                    attempts += 1
                    if fixture:
                        write_flac(output, fixture_voice(turn["text"], plan_index + attempts))
                    else:
                        _run_adapter(synth_command or "", {**recipe, "attempt": attempts}, output)
                    metrics = quality_metrics(output)
                    if metrics["duration_seconds"] < 0.3 or metrics["duration_seconds"] > 15:
                        raise ValueError(f"synthesized utterance duration is invalid: {output}")
                    if not metrics["finite"] or metrics["clipping_fraction"] > 0.001:
                        raise ValueError(f"synthesized utterance quality is invalid: {output}")
                    metric, score = _asr_score(
                        asr_command, output, turn["text"], language, fixture
                    )
                    if score <= 0.20 or attempts == 2:
                        break
            if score > 0.20:
                output.unlink(missing_ok=True)
                rejected += 1
                continue
            receipt = {
                **recipe,
                "recipe_sha256": recipe_hash,
                "audio": relative_to_root(output, root),
                "audio_sha256": sha256_file(output),
                "duration_seconds": metrics["duration_seconds"],
                "asr_metric": metric,
                "asr_score": score,
                "attempts": attempts,
                "fixture": fixture,
            }
            if not fixture:
                receipt["normalization"] = _formal_audio_report(output)
                if not cached:
                    _store_synthesis_cache(
                        root,
                        recipe,
                        output,
                        metric=metric,
                        score=score,
                        attempts=attempts,
                    )
            write_json(receipt_path, receipt)
            receipts.append(receipt)
    source_inventory = registry_path(root, "normalized", "source-items.jsonl")
    if source_inventory.exists():
        from data.curation.common import read_jsonl

        for source_index, item in enumerate(read_jsonl(source_inventory)):
            if item.get("category") != "adjacent_turns" or not item.get("response_text"):
                continue
            language = str(item["language"])
            text = str(item["response_text"])
            turn_id = "assistant-response"
            recipe = {
                "source_item_id": item["source_item_id"],
                "turn_id": turn_id,
                "text": text,
                "language": language,
                "role": "assistant",
                "voice_id": assistant_voice,
                "voice_prompt_sha256": voices[assistant_voice]["prompt_sha256"],
                "voice_prompt_audio": str(
                    (root / str(voices[assistant_voice]["prompt_audio"])).resolve()
                ),
                "voice_prompt_text": str(
                    voices[assistant_voice].get("prompt_text") or ""
                ),
                "model_sha256": model_hash,
            }
            recipe_hash = stable_hash(recipe)
            output = dataset_path(
                root,
                dataset,
                "synthesized",
                "source-responses",
                f"{item['source_item_id']}.flac",
            )
            receipt_path = output.with_suffix(".json")
            if output.is_file() and receipt_path.is_file():
                old = read_json(receipt_path)
                if old.get("recipe_sha256") == recipe_hash and old.get(
                    "audio_sha256"
                ) == sha256_file(output):
                    receipts.append(old)
                    continue
            cached = None if fixture else _cached_synthesis(root, recipe, output)
            if cached:
                cache_receipt, _ = cached
                attempts = int(cache_receipt["attempts"])
                metric = str(cache_receipt["asr_metric"])
                score = float(cache_receipt["asr_score"])
                metrics = quality_metrics(output)
            else:
                attempts = 0
                while True:
                    attempts += 1
                    if fixture:
                        write_flac(output, fixture_voice(text, source_index + attempts))
                    else:
                        _run_adapter(synth_command or "", {**recipe, "attempt": attempts}, output)
                    metrics = quality_metrics(output)
                    if metrics["duration_seconds"] < 0.3 or metrics["duration_seconds"] > 15:
                        raise ValueError(f"synthesized utterance duration is invalid: {output}")
                    if not metrics["finite"] or metrics["clipping_fraction"] > 0.001:
                        raise ValueError(f"synthesized utterance quality is invalid: {output}")
                    metric, score = _asr_score(asr_command, output, text, language, fixture)
                    if score <= 0.20 or attempts == 2:
                        break
            if score > 0.20:
                output.unlink(missing_ok=True)
                rejected += 1
                continue
            receipt = {
                **recipe,
                "recipe_sha256": recipe_hash,
                "audio": relative_to_root(output, root),
                "audio_sha256": sha256_file(output),
                "duration_seconds": metrics["duration_seconds"],
                "asr_metric": metric,
                "asr_score": score,
                "attempts": attempts,
                "fixture": fixture,
            }
            if not fixture:
                receipt["normalization"] = _formal_audio_report(output)
                if not cached:
                    _store_synthesis_cache(
                        root,
                        recipe,
                        output,
                        metric=metric,
                        score=score,
                        attempts=attempts,
                    )
            write_json(receipt_path, receipt)
            receipts.append(receipt)
    path = dataset_path(root, dataset, "synthesized", "utterances.jsonl")
    write_jsonl(path, receipts)
    report = {
        "dataset": dataset,
        "fixture": fixture,
        "utterances": len(receipts),
        "rejected": rejected,
        "manifest": str(path),
        "manifest_sha256": sha256_file(path),
        "tts_model_sha256": model_hash,
        "prompt_registry_sha256": sha256_file(voice_registry_path),
    }
    write_json(dataset_path(root, dataset, "reports", "synthesis.json"), report)
    return report
