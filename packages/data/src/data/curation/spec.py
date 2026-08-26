from __future__ import annotations

from dataclasses import dataclass

from data.curation.common import CATEGORIES, LANGUAGES, SPLITS

CATEGORY_FRACTIONS = {
    "public_speech": 0.30,
    "synthetic_dialogue": 0.50,
    "adjacent_turns": 0.15,
    "screen_task": 0.05,
}
LANGUAGE_FRACTIONS = {"zh": 0.80, "en": 0.20}
SPLIT_FRACTIONS = {"train": 0.80, "validation": 0.10, "test": 0.10}
DATASET_HOURS = {"canary": 1.0}
MIMI_CODEC_ID = "mimi-24khz-8x2048"
MIMI_REVISION = "a49141e28b3d9c947cf9aa5314431e1b11cbd2f5"
MIMI_WEIGHT_SHA256 = "09b782f0629851a271227fb9d36db65c041790365f11bbe5d3d59369cf863f50"

SOURCE_CATALOG = {
    "aishell1": {
        "source_version": "hf-AISHELL-1@bbe295d5",
        "source_url": "https://huggingface.co/datasets/AISHELL/AISHELL-1",
        "license": "Apache-2.0",
        "license_url": "https://www.openslr.org/33/",
        "language": "zh",
        "category": "public_speech",
    },
    "librispeech_train_clean_100": {
        "source_version": "hf-librispeech_asr@71cacbfb",
        "source_url": "https://huggingface.co/datasets/openslr/librispeech_asr",
        "license": "CC-BY-4.0",
        "license_url": "https://www.openslr.org/12/",
        "language": "en",
        "category": "public_speech",
    },
    "aishell4_train_l": {
        "source_version": "hf-argmaxinc-aishell-4@0cf6e538",
        "source_url": "https://huggingface.co/datasets/argmaxinc/aishell-4",
        "license": "CC-BY-SA-4.0",
        "license_url": "https://www.openslr.org/111/",
        "language": "zh",
        "category": "adjacent_turns",
    },
    "dailytalk": {
        "source_version": "hf-DailyTalk@1f0d958a",
        "source_url": (
            "https://huggingface.co/datasets/"
            "DynamicSuperbPrivate/DialogueActClassification_DailyTalk"
        ),
        "license": "CC-BY-SA-4.0 AND academic-use-statement",
        "license_url": "https://github.com/keonlee9420/DailyTalk",
        "language": "en",
        "category": "adjacent_turns",
    },
}


@dataclass(frozen=True, slots=True)
class DatasetSpec:
    name: str
    duration_seconds: float

    def target_seconds(self, category: str, language: str, split: str) -> float:
        return (
            self.duration_seconds
            * CATEGORY_FRACTIONS[category]
            * LANGUAGE_FRACTIONS[language]
            * SPLIT_FRACTIONS[split]
        )

    def quota_rows(self) -> list[dict[str, object]]:
        return [
            {
                "category": category,
                "language": language,
                "split": split,
                "target_seconds": self.target_seconds(category, language, split),
            }
            for category in CATEGORIES
            for language in LANGUAGES
            for split in SPLITS
        ]


def dataset_spec(name: str) -> DatasetSpec:
    if name not in DATASET_HOURS:
        raise ValueError(f"dataset must be one of {sorted(DATASET_HOURS)}")
    return DatasetSpec(name, DATASET_HOURS[name] * 3600)
