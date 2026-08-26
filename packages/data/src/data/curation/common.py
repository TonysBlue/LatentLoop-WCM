from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Iterable
from pathlib import Path
from typing import Any

DATASETS = ("canary",)
SPLITS = ("train", "validation", "test")
CATEGORIES = ("public_speech", "synthetic_dialogue", "adjacent_turns", "screen_task")
LANGUAGES = ("zh", "en")


def data_root(value: str | Path) -> Path:
    return Path(value).expanduser().resolve()


def asset_root(root: str | Path) -> Path:
    root = Path(root).expanduser().resolve()
    value = os.environ.get("LATENTLOOP_ASSET_ROOT", root.parent / "assets")
    return Path(value).expanduser().resolve()


def source_root(root: str | Path) -> Path:
    return asset_root(root) / "sources"


def registry_root(root: str | Path) -> Path:
    return Path(root).expanduser().resolve() / "registry"


def registry_path(root: str | Path, *parts: str) -> Path:
    return registry_root(root).joinpath(*parts)


def dataset_root(root: str | Path, dataset: str) -> Path:
    return Path(root).expanduser().resolve() / dataset


def dataset_path(root: str | Path, dataset: str, *parts: str) -> Path:
    return dataset_root(root, dataset).joinpath(*parts)


def ensure_tree(root: Path) -> None:
    registry_root(root).mkdir(parents=True, exist_ok=True)
    for relative in ("licenses", "normalized", "voices", "reports", "synthesis-cache"):
        registry_path(root, relative).mkdir(parents=True, exist_ok=True)
    for dataset in DATASETS:
        dataset_root(root, dataset).mkdir(parents=True, exist_ok=True)
        for relative in ("text", "synthesized", "normalized", "manifests", "reports"):
            dataset_path(root, dataset, relative).mkdir(parents=True, exist_ok=True)
        for split in SPLITS:
            dataset_path(root, dataset, "shards", "staging", split).mkdir(
                parents=True, exist_ok=True
            )
            dataset_path(root, dataset, "shards", "processed", split).mkdir(
                parents=True, exist_ok=True
            )


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_hash(value: Any) -> str:
    encoded = json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )
    return hashlib.sha256(encoded).hexdigest()


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with Path(path).open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number} must contain a JSON object")
            records.append(value)
    return records


def _atomic_bytes(path: Path, value: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as output:
            output.write(value)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def write_json(path: str | Path, value: Any) -> None:
    encoded = (json.dumps(value, indent=2, ensure_ascii=True, sort_keys=True) + "\n").encode()
    _atomic_bytes(Path(path), encoded)


def write_jsonl(path: str | Path, records: Iterable[dict[str, Any]]) -> None:
    encoded = b"".join(
        (json.dumps(record, ensure_ascii=True, sort_keys=True) + "\n").encode()
        for record in records
    )
    _atomic_bytes(Path(path), encoded)


def require_sha256(value: object, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError(f"{label} must be a 64-character SHA-256 digest")
    try:
        int(value, 16)
    except ValueError as error:
        raise ValueError(f"{label} must be a hexadecimal SHA-256 digest") from error
    return value.lower()


def assign_split(group_id: str, seed: int = 17) -> str:
    bucket = int(stable_hash({"group": group_id, "seed": seed})[:8], 16) % 10
    if bucket < 8:
        return "train"
    return "validation" if bucket == 8 else "test"


def relative_to_root(path: Path, root: Path) -> str:
    return str(path.resolve().relative_to(root.resolve()))
