from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

from data.curation.audio import fixture_voice, write_flac
from data.curation.common import (
    ensure_tree,
    read_json,
    registry_path,
    relative_to_root,
    sha256_file,
    stable_hash,
    write_json,
)


def _validate_voice(record: dict[str, Any], root: Path) -> dict[str, Any]:
    required = ("voice_id", "language", "license", "authorization", "prompt_audio")
    missing = [field for field in required if not record.get(field)]
    if missing:
        raise ValueError(f"voice record is missing {missing}")
    path = Path(record["prompt_audio"]).expanduser()
    if not path.is_absolute():
        path = root / path
    if not path.is_file():
        raise FileNotFoundError(f"voice prompt is absent: {path}")
    actual = sha256_file(path)
    if record.get("prompt_sha256") not in {None, actual}:
        raise ValueError(f"voice prompt hash mismatch for {record['voice_id']}")
    suffix = path.suffix.lower()
    if suffix not in {".flac", ".wav"}:
        raise ValueError("voice prompt must be FLAC or WAV")
    stored = registry_path(root, "voices", "prompts", f"{record['voice_id']}-{actual[:12]}{suffix}")
    stored.parent.mkdir(parents=True, exist_ok=True)
    if not stored.exists() or sha256_file(stored) != actual:
        shutil.copy2(path, stored)
    return {
        **record,
        "prompt_audio": relative_to_root(stored, root),
        "prompt_sha256": actual,
    }


def select_canary_voices(
    root: str | Path,
    *,
    library: str | Path | None = None,
    fixture: bool = False,
) -> dict[str, Any]:
    root = Path(root).expanduser().resolve()
    ensure_tree(root)
    if fixture:
        voices = []
        voice_specs = [("assistant-neutral", "multilingual", "assistant", None)]
        voice_specs.extend(
            (f"user-{language}-{split}", language, "user", split)
            for language in ("zh", "en")
            for split in ("train", "validation", "test")
        )
        for index, (voice_id, language, role, split) in enumerate(voice_specs):
            prompt = registry_path(root, "voices", "fixture", f"{voice_id}.flac")
            write_flac(prompt, fixture_voice(voice_id, index))
            voices.append(
                {
                    "voice_id": voice_id,
                    "language": language,
                    "role": role,
                    "split": split,
                    "license": "fixture-only",
                    "authorization": "not for model training",
                    "prompt_audio": relative_to_root(prompt, root),
                    "prompt_sha256": sha256_file(prompt),
                    "fixture": True,
                }
            )
    else:
        if library is None:
            raise ValueError(
                "formal Canary voice selection requires --library with CosyVoice example "
                "voice records"
            )
        raw = read_json(Path(library).expanduser())
        if not isinstance(raw, list):
            raise ValueError("voice library must be a JSON list")
        voices = [_validate_voice(record, root) for record in raw]
    assistants = [voice for voice in voices if voice.get("role") == "assistant"]
    if len(assistants) != 1:
        raise ValueError("voice registry requires exactly one fixed assistant voice")
    for language in ("zh", "en"):
        if not any(
            voice.get("role") == "user" and voice.get("language") in {language, "multilingual"}
            for voice in voices
        ):
            raise ValueError(f"voice registry has no {language} user voice")
    user_voice_splits: dict[str, set[str]] = {}
    for voice in voices:
        if voice.get("role") != "user":
            continue
        split = voice.get("split")
        if split not in {"train", "validation", "test"}:
            raise ValueError("each user voice must be assigned to exactly one dataset split")
        user_voice_splits.setdefault(str(voice["voice_id"]), set()).add(str(split))
    if any(len(splits) != 1 for splits in user_voice_splits.values()):
        raise ValueError("a user voice cannot cross dataset splits")
    registry = {
        "assistant_voice_id": assistants[0]["voice_id"],
        "voices": voices,
    }
    registry["registry_sha256"] = stable_hash(registry)
    path = registry_path(root, "voices", "registry.json")
    write_json(path, registry)
    return {"path": str(path), "voices": len(voices), "sha256": sha256_file(path)}
