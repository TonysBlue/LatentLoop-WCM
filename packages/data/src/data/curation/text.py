from __future__ import annotations

import random
from pathlib import Path
from typing import Any

from data.curation.common import (
    dataset_path,
    ensure_tree,
    sha256_file,
    stable_hash,
    write_json,
)
from data.curation.spec import (
    LANGUAGE_FRACTIONS,
    SPLIT_FRACTIONS,
    dataset_spec,
)

INTENTS = {
    "browser_search": 20,
    "file_management": 15,
    "document_editing": 15,
    "apps_windows": 15,
    "system_settings": 10,
    "state_reading": 10,
    "confirmation": 10,
    "safe_recovery": 5,
}

ZH_COPY = {
    "browser_search": ("搜索今天的天气。", "我正在查看天气信息。", "结果已经显示。"),
    "file_management": ("找到下载目录里的报告。", "我正在检查下载目录。", "已经找到报告。"),
    "document_editing": ("在文档末尾加入会议日期。", "我会先定位文档末尾。", "日期已经加入。"),
    "apps_windows": ("切换到浏览器窗口。", "我正在查看当前窗口。", "已经切换到浏览器。"),
    "system_settings": ("把系统音量调低一些。", "我会先读取当前音量。", "音量已经调低。"),
    "state_reading": ("告诉我当前打开了哪些应用。", "我正在查看任务栏。", "我已经读到应用列表。"),
    "confirmation": (
        "关闭这个未保存的窗口。",
        "窗口包含未保存内容，是否继续关闭？",
        "我会等待你的确认。",
    ),
    "safe_recovery": (
        "把所有文件永久删除。",
        "这个操作不可逆，我不会直接执行。",
        "可以先建立备份再处理。",
    ),
}

_FORMAL_DURATION_COUNTS = {
    ("zh", "train"): (48, 19, 9),
    ("zh", "validation"): (6, 3, 1),
    ("zh", "test"): (6, 2, 2),
    ("en", "train"): (9, 4, 6),
    ("en", "validation"): (1, 1, 0),
    ("en", "test"): (2, 1, 0),
}
EN_COPY = {
    "browser_search": (
        "Search for today's weather.",
        "I am checking the weather.",
        "The results are visible.",
    ),
    "file_management": (
        "Find the report in Downloads.",
        "I am looking for the report now.",
        "I found the report.",
    ),
    "document_editing": (
        "Add the meeting date to the document.",
        "I will locate the end first.",
        "The date is now added.",
    ),
    "apps_windows": (
        "Switch to the browser window.",
        "I am checking the open windows.",
        "The browser is active.",
    ),
    "system_settings": (
        "Turn the system volume down.",
        "I will read the current level.",
        "The volume is lower.",
    ),
    "state_reading": (
        "Tell me which applications are open.",
        "I am inspecting the taskbar.",
        "I can now read the list.",
    ),
    "confirmation": (
        "Close this unsaved window.",
        "It has unsaved work. Should I continue?",
        "I will wait for confirmation.",
    ),
    "safe_recovery": (
        "Permanently delete every file.",
        "That is irreversible, so I will not run it directly.",
        "We can create a backup first.",
    ),
}


def _weighted_intents(count: int) -> list[str]:
    values: list[str] = []
    for intent, percent in INTENTS.items():
        values.extend([intent] * (count * percent // 100))
    while len(values) < count:
        values.append(list(INTENTS)[len(values) % len(INTENTS)])
    return values[:count]


def plan_recipe_sha256(plan: dict[str, Any]) -> str:
    return stable_hash(
        {key: value for key, value in plan.items() if key != "recipe_sha256"}
    )


def _split(index: int, count: int) -> str:
    if index < count * 8 // 10:
        return "train"
    return "validation" if index < count * 9 // 10 else "test"


def _duration_bounds(plan: dict[str, Any]) -> tuple[float, float]:
    return {
        "short": (4.0, 16.0),
        "medium": (16.0, 32.0),
        "long": (32.0, 60.0),
    }[str(plan["duration_class"])]


def _allocate_duration(plans: list[dict[str, Any]], target: float) -> None:
    bounds = [_duration_bounds(plan) for plan in plans]
    minimum = sum(lower for lower, _ in bounds)
    maximum = sum(upper for _, upper in bounds)
    if not minimum <= target <= maximum:
        raise ValueError(
            f"plan duration capacity [{minimum}, {maximum}] cannot cover {target} seconds"
        )
    capacities = [upper - lower for lower, upper in bounds]
    remaining = target - minimum
    total_capacity = sum(capacities)
    durations = [
        lower + (remaining * capacity / total_capacity if total_capacity else 0.0)
        for (lower, _), capacity in zip(bounds, capacities, strict=True)
    ]
    correction = target - sum(durations)
    durations[-1] += correction
    for plan, duration in zip(plans, durations, strict=True):
        plan["target_duration_seconds"] = round(duration, 6)


def _shift_screen_duration(plans: list[dict[str, Any]], target: float) -> None:
    current = sum(
        float(plan["target_duration_seconds"])
        for plan in plans
        if plan["category"] == "screen_task"
    )
    delta = target - current
    if abs(delta) < 1e-6:
        return
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for plan in plans:
        groups.setdefault((str(plan["language"]), str(plan["split"])), []).append(plan)
    for group in groups.values():
        screen = [plan for plan in group if plan["category"] == "screen_task"]
        dialogue = [plan for plan in group if plan["category"] == "synthetic_dialogue"]
        if delta > 0:
            receivers, donors = screen, dialogue
        else:
            receivers, donors = dialogue, screen
        receiver_capacity = sum(
            _duration_bounds(plan)[1] - float(plan["target_duration_seconds"]) for plan in receivers
        )
        donor_capacity = sum(
            float(plan["target_duration_seconds"]) - _duration_bounds(plan)[0] for plan in donors
        )
        moved = min(abs(delta), receiver_capacity, donor_capacity)
        if moved <= 0:
            continue
        left = moved
        for plan in receivers:
            capacity = _duration_bounds(plan)[1] - float(plan["target_duration_seconds"])
            increment = min(left, capacity)
            plan["target_duration_seconds"] = float(plan["target_duration_seconds"]) + increment
            left -= increment
            if left <= 1e-9:
                break
        left = moved
        for plan in donors:
            capacity = float(plan["target_duration_seconds"]) - _duration_bounds(plan)[0]
            decrement = min(left, capacity)
            plan["target_duration_seconds"] = float(plan["target_duration_seconds"]) - decrement
            left -= decrement
            if left <= 1e-9:
                break
        delta += -moved if delta > 0 else moved
        if abs(delta) <= 1e-6:
            break
    if abs(delta) > 1e-5:
        raise ValueError("plan mix cannot reach the screen-task duration quota")
    for plan in plans:
        plan["target_duration_seconds"] = round(float(plan["target_duration_seconds"]), 6)


def _select_screen_plans(
    plans: list[dict[str, Any]], screen_target: float, group_target: float
) -> tuple[set[str], float]:
    by_class = {
        duration_class: sorted(
            (plan for plan in plans if plan["duration_class"] == duration_class),
            key=lambda plan: stable_hash(plan["plan_id"]),
        )
        for duration_class in ("short", "medium", "long")
    }
    desired_count = max(1, round(len(plans) / 11), int(-(-screen_target // 60)))
    total_lower = sum(_duration_bounds(plan)[0] for plan in plans)
    total_upper = sum(_duration_bounds(plan)[1] for plan in plans)
    best: tuple[tuple[float, int, float], tuple[int, int, int], float] | None = None
    candidate_counts = sorted(range(1, len(plans)), key=lambda count: abs(count - desired_count))
    for count in candidate_counts:
        minimum_short = max(0, count - len(by_class["medium"]) - len(by_class["long"]))
        maximum_short = min(count, len(by_class["short"]))
        for short_count in range(minimum_short, maximum_short + 1):
            medium_count = count - short_count
            minimum_medium = max(0, medium_count - len(by_class["long"]))
            maximum_medium = min(medium_count, len(by_class["medium"]))
            for selected_medium in range(minimum_medium, maximum_medium + 1):
                long_count = count - short_count - selected_medium
                lower = 4 * short_count + 16 * selected_medium + 32 * long_count
                upper = 16 * short_count + 32 * selected_medium + 60 * long_count
                dialogue_lower = total_lower - lower
                dialogue_upper = total_upper - upper
                feasible_lower = max(lower, group_target - dialogue_upper)
                feasible_upper = min(upper, group_target - dialogue_lower)
                if feasible_lower > feasible_upper:
                    continue
                attainable = min(max(screen_target, feasible_lower), feasible_upper)
                score = (
                    abs(attainable - screen_target),
                    abs(count - desired_count),
                    abs((feasible_lower + feasible_upper) / 2 - screen_target),
                )
                if best is None or score < best[0]:
                    best = (
                        score,
                        (short_count, selected_medium, long_count),
                        attainable,
                    )
        if best is not None and best[0][0] <= 1e-9 and best[0][1] == 0:
            break
    if best is None:
        raise ValueError("no duration-class assignment can satisfy the screen quota")
    counts = best[1]
    selected = {
        plan["plan_id"]
        for duration_class, count in zip(("short", "medium", "long"), counts, strict=True)
        for plan in by_class[duration_class][:count]
    }
    return selected, best[2]


def _calibrate_plan_mix(plans: list[dict[str, Any]], dataset: str) -> None:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for plan in plans:
        groups.setdefault((str(plan["language"]), str(plan["split"])), []).append(plan)
    total_seconds = dataset_spec(dataset).duration_seconds
    for (language, split), group in groups.items():
        requested_screen = (
            total_seconds * 0.05 * LANGUAGE_FRACTIONS[language] * SPLIT_FRACTIONS[split]
        )
        group_target = total_seconds * 0.55 * LANGUAGE_FRACTIONS[language] * SPLIT_FRACTIONS[split]
        screen_ids, attainable_screen = _select_screen_plans(group, requested_screen, group_target)
        for plan in group:
            plan["category"] = (
                "screen_task" if plan["plan_id"] in screen_ids else "synthetic_dialogue"
            )
        _allocate_duration(group, group_target)
        _shift_screen_duration(group, attainable_screen)
    _shift_screen_duration(plans, total_seconds * 0.05)
    for plan in plans:
        plan["recipe_sha256"] = plan_recipe_sha256(plan)


def _plan(
    dataset: str,
    index: int,
    duration_class: str,
    language: str,
    intent: str,
    split: str,
    fixture: bool,
    plan_offset: int,
) -> dict[str, Any]:
    copy = (ZH_COPY if language == "zh" else EN_COPY)[intent]
    multi = index % 20 >= 12
    turns: list[dict[str, str]] = [
        {"turn_id": "turn-00", "role": "user", "text": copy[0]},
        {"turn_id": "turn-01", "role": "assistant", "text": copy[1]},
    ]
    if multi:
        turns.extend(
            [
                {
                    "turn_id": "turn-02",
                    "role": "user",
                    "text": "继续。" if language == "zh" else "Continue.",
                },
                {"turn_id": "turn-03", "role": "assistant", "text": copy[2]},
                {
                    "turn_id": "turn-04",
                    "role": "user",
                    "text": "好的。" if language == "zh" else "Okay.",
                },
                {"turn_id": "turn-05", "role": "assistant", "text": copy[2]},
            ]
        )
    duration_seconds = {
        "short": 4 + index % 9,
        "medium": 16 + index % 9,
        "long": 32 + index % 9,
    }[duration_class]
    category = "screen_task" if fixture and index in {0, 9, 10} else "synthetic_dialogue"
    # IDs are stable within the Canary data catalog.
    plan_id = f"plan-{language}-{plan_offset + index:04d}"
    template_id = f"{dataset}-{intent}-{split}-v{index % 7}"
    scenario_id = f"{dataset}-{split}-{intent}-{index:04d}"
    value = {
        "plan_id": plan_id,
        "dataset": dataset,
        "category": category,
        "language": language,
        "split": split,
        "intent": intent,
        "template_id": template_id,
        "scenario_id": scenario_id,
        "duration_class": duration_class,
        "target_duration_seconds": duration_seconds,
        "turns": turns,
        # Text plans are generated deterministically and admitted by machine
        # gates downstream; there is no manual approval state in this pipeline.
        "quality": {
            "status": "generated",
            "generator": "latentloop-canary-text-v1",
            "fixture": fixture,
        },
    }
    value["recipe_sha256"] = plan_recipe_sha256(value)
    return value


def build_canary_text(
    root: str | Path,
    *,
    fixture: bool = False,
    seed: int = 17,
) -> dict[str, Any]:
    dataset = "canary"
    root = Path(root).expanduser().resolve()
    ensure_tree(root)
    if fixture:
        language_counts = {"zh": 12, "en": 12}
    else:
        language_counts = {"zh": 96, "en": 24}
    plans: list[dict[str, Any]] = []
    duration_offsets: dict[tuple[str, str], int] = {}
    if fixture:
        offsets = {"canary": {"zh": 0, "en": 0}}
    else:
        offsets = {"canary": {"zh": 0, "en": 0}}
    for language, count in language_counts.items():
        intents = _weighted_intents(count)
        random.Random(seed + (0 if language == "zh" else 1)).shuffle(intents)
        for index in range(count):
            split = _split(index, count)
            duration_offset = duration_offsets.get((language, split), 0)
            if fixture:
                duration_class = "short"
            else:
                short_count, medium_count, _ = _FORMAL_DURATION_COUNTS[(language, split)]
                duration_class = (
                    "short"
                    if duration_offset < short_count
                    else "medium"
                    if duration_offset < short_count + medium_count
                    else "long"
                )
            duration_offsets[(language, split)] = duration_offset + 1
            plans.append(
                _plan(
                    dataset,
                    index,
                    duration_class,
                    language,
                    intents[index],
                    split,
                    fixture,
                    offsets[dataset][language],
                )
            )
    if not fixture:
        _calibrate_plan_mix(plans, dataset)
    path = dataset_path(root, dataset, "text", "plans.json")
    value = {
        "dataset": dataset,
        "fixture": fixture,
        "seed": seed,
        "plans": plans,
    }
    write_json(path, value)
    assistant_responses = sum(
        turn["role"] == "assistant" for plan in plans for turn in plan["turns"]
    )
    report = {
        "path": str(path),
        "sha256": sha256_file(path),
        "plans": len(plans),
        "assistant_responses": assistant_responses,
        "languages": language_counts,
        "quality_status": "generated",
    }
    write_json(dataset_path(root, dataset, "reports", "text.json"), report)
    return report
