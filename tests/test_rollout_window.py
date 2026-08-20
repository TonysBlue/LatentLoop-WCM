from __future__ import annotations

import pytest
from training.window import RolloutWindowStore, SealedRolloutWindow


def test_rollout_window_store_is_immutable_and_finalized(tmp_path) -> None:
    window = SealedRolloutWindow(
        "window", "life", "session", "policy", 10, 19, "chain", 19, ("event",), 10,
        20, "a" * 64,
    )
    store = RolloutWindowStore(tmp_path)
    path, digest = store.seal(window)
    assert path.is_file()
    assert len(digest) == 64
    with pytest.raises(ValueError, match="already sealed"):
        store.seal(window)


def test_rollout_window_rejects_unfinalized_reward() -> None:
    with pytest.raises(ValueError, match="not finalized"):
        SealedRolloutWindow(
            "window", "life", "session", "policy", 10, 19, "chain", 18, (), 10,
            20, "a" * 64,
        )


def test_rollout_window_requires_exact_hashed_lookahead() -> None:
    with pytest.raises(ValueError, match="immediate successor"):
        SealedRolloutWindow(
            "window", "life", "session", "policy", 10, 19, "chain", 19, (), 10,
            21, "a" * 64,
        )
