from __future__ import annotations

import shutil
import tarfile
import urllib.request
from pathlib import Path
from typing import Any

from data.curation.audio import fixture_voice, write_flac
from data.curation.common import (
    ensure_tree,
    read_json,
    registry_path,
    relative_to_root,
    require_sha256,
    sha256_file,
    source_root,
    stable_hash,
    write_json,
    write_jsonl,
)
from data.curation.spec import SOURCE_CATALOG


def _safe_extract(archive: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive) as bundle:
        root = destination.resolve()
        for member in bundle.getmembers():
            target = (destination / member.name).resolve()
            if root not in target.parents and target != root:
                raise ValueError(f"archive contains an unsafe path: {member.name}")
        bundle.extractall(destination, filter="data")


def _is_tar_archive(path: Path) -> bool:
    return tarfile.is_tarfile(path)


def _download(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".partial")
    request = urllib.request.Request(url, headers={"User-Agent": "LatentLoop-Canary/0.1"})
    with urllib.request.urlopen(request, timeout=60) as response, temporary.open("wb") as output:
        shutil.copyfileobj(response, output)
    temporary.replace(destination)


def _fixture(root: Path) -> dict[str, Any]:
    license_records = []
    for source_id, source in SOURCE_CATALOG.items():
        path = registry_path(root, "licenses", f"{source_id}.txt")
        path.write_text(
            f"Fixture-only license record for {source_id}: {source['license']}\n",
            encoding="utf-8",
        )
        license_records.append(
            {
                "source_id": source_id,
                "license": source["license"],
                "license_path": relative_to_root(path, root),
                "license_sha256": sha256_file(path),
                "fixture": True,
            }
        )
    records = []
    fixture_dir = registry_path(root, "normalized", "fixture-sources")
    mapping = {
        ("public_speech", "zh"): "aishell1",
        ("public_speech", "en"): "librispeech_train_clean_100",
        ("adjacent_turns", "zh"): "aishell4_train_l",
        ("adjacent_turns", "en"): "dailytalk",
    }
    texts = {
        "zh": ("请查看当前窗口。", "当前窗口已经打开。"),
        "en": ("Please inspect the current window.", "The current window is open."),
    }
    for (category, language), source_id in mapping.items():
        source = SOURCE_CATALOG[source_id]
        license_record = next(item for item in license_records if item["source_id"] == source_id)
        for split in ("train", "validation", "test"):
            for copy_index in range(2):
                item_id = f"fixture-{category}-{language}-{split}-{copy_index}"
                audio = fixture_dir / f"{item_id}.flac"
                write_flac(audio, fixture_voice(texts[language][0], len(records)))
                records.append(
                    {
                        "source_item_id": item_id,
                        "source_id": source_id,
                        "source_version": source["source_version"],
                        "source_url": source["source_url"] or source["license_url"],
                        "source_utterance_ids": [item_id],
                        "source_license": source["license"],
                        "license_sha256": license_record["license_sha256"],
                        "redistribution_allowed": True,
                        "category": category,
                        "language": language,
                        "split": split,
                        "group_id": item_id,
                        "speaker_id": f"speaker-{item_id}",
                        "session_id": f"session-{item_id}",
                        "text": texts[language][0],
                        "response_text": texts[language][1] if category == "adjacent_turns" else "",
                        "audio": relative_to_root(audio, root),
                        "audio_sha256": sha256_file(audio),
                        "normalization": {
                            "integrated_lufs": None,
                            "metrics_sha256": "fixture",
                        },
                        "fixture": True,
                    }
                )
    inventory = registry_path(root, "normalized", "source-items.jsonl")
    write_jsonl(inventory, records)
    report = {
        "fixture": True,
        "sources": len(SOURCE_CATALOG),
        "items": len(records),
        "inventory_sha256": sha256_file(inventory),
        "license_records": license_records,
    }
    write_json(registry_path(root, "reports", "fetch-report.json"), report)
    return report


def fetch_canary_data(
    root: str | Path,
    *,
    fixture: bool = False,
    lock_path: str | Path | None = None,
    download: bool = False,
    extract: bool = False,
) -> dict[str, Any]:
    root = Path(root).expanduser().resolve()
    ensure_tree(root)
    if fixture:
        return _fixture(root)
    if lock_path is None:
        template = {
            source_id: {
                **source,
                "archive_sha256": "<required>",
                "license_path": "<required local license record>",
                "license_sha256": "<required>",
            }
            for source_id, source in SOURCE_CATALOG.items()
        }
        write_json(registry_path(root, "source-lock.template.json"), template)
        raise ValueError(
            "formal Canary fetch requires --lock; a source-lock template was written "
            "under registry/"
        )
    lock = read_json(Path(lock_path).expanduser())
    if set(lock) != set(SOURCE_CATALOG):
        raise ValueError("source lock must contain exactly the locked Canary source catalog")
    results = []
    for source_id, expected in SOURCE_CATALOG.items():
        record = lock[source_id]
        if record.get("source_version") != expected["source_version"]:
            raise ValueError(f"{source_id} source_version differs from the locked catalog")
        expected_license = require_sha256(record.get("license_sha256"), f"{source_id} license")
        license_path = Path(record["license_path"]).expanduser().resolve()
        if not license_path.is_file() or sha256_file(license_path) != expected_license:
            raise ValueError(f"{source_id} license record is missing or has the wrong hash")
        stored_license = registry_path(
            root, "licenses", f"{source_id}{license_path.suffix or '.txt'}"
        )
        shutil.copy2(license_path, stored_license)
        archives = record.get("archives")
        if archives is None:
            archives = [record]
        if not isinstance(archives, list) or not archives:
            raise ValueError(f"{source_id} archives must be a non-empty list")
        for archive_index, archive in enumerate(archives):
            url = str(archive.get("source_url") or "")
            if not url:
                raise ValueError(f"{source_id} archive {archive_index} has no URL")
            expected_archive = require_sha256(
                archive.get("archive_sha256"), f"{source_id} archive {archive_index}"
            )
            filename = str(archive.get("filename") or Path(url).name)
            destination = source_root(root) / source_id / filename
            if not destination.exists():
                if not download:
                    raise FileNotFoundError(
                        f"{destination} is absent; rerun with --download after reviewing the lock"
                    )
                _download(url, destination)
            actual_archive = sha256_file(destination)
            if actual_archive != expected_archive:
                raise ValueError(f"{source_id} archive {archive_index} SHA-256 mismatch")
            if extract and _is_tar_archive(destination):
                _safe_extract(destination, source_root(root) / source_id / "extracted")
            results.append(
                {
                    "source_id": source_id,
                    "source_version": expected["source_version"],
                    "source_url": url,
                    "archive": str(destination.resolve()),
                    "archive_sha256": actual_archive,
                    "license": expected["license"],
                    "license_path": relative_to_root(stored_license, root),
                    "license_sha256": expected_license,
                    "recipe_sha256": stable_hash(record),
                }
            )
    report = {"fixture": False, "sources": results}
    write_json(registry_path(root, "reports", "fetch-report.json"), report)
    return report
