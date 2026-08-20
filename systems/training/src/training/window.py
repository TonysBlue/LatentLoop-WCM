from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class SealedRolloutWindow:
    window_id: str
    lineage_id: str
    session_id: str
    policy_version: str
    start_unit: int
    end_unit: int
    observation_chain_sha256: str
    finalized_through_unit: int
    reward_event_ids: tuple[str, ...]
    consumed_units: int
    lookahead_unit_index: int
    lookahead_payload_sha256: str
    eligible_for_update: bool = True
    disposition: str = "trainable"

    def __post_init__(self) -> None:
        identities = (
            self.window_id,
            self.lineage_id,
            self.session_id,
            self.policy_version,
            self.observation_chain_sha256,
        )
        if not all(identities):
            raise ValueError("sealed PPO window identity is incomplete")
        if self.start_unit < 0 or self.end_unit < self.start_unit:
            raise ValueError("sealed PPO window unit range is invalid")
        if self.consumed_units != self.end_unit - self.start_unit + 1:
            raise ValueError("sealed PPO window unit count is invalid")
        if self.lookahead_unit_index != self.end_unit + 1:
            raise ValueError("sealed PPO window lookahead must be the immediate successor")
        if len(self.lookahead_payload_sha256) != 64 or any(
            character not in "0123456789abcdef"
            for character in self.lookahead_payload_sha256
        ):
            raise ValueError("sealed PPO window lookahead payload hash is invalid")
        if not self.disposition:
            raise ValueError("sealed PPO window disposition is required")
        if self.eligible_for_update and self.disposition != "trainable":
            raise ValueError("trainable PPO windows require trainable disposition")
        if self.eligible_for_update and self.finalized_through_unit < self.end_unit:
            raise ValueError("sealed PPO window reward is not finalized")


class RolloutWindowStore:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def seal(self, window: SealedRolloutWindow) -> tuple[Path, str]:
        target = self.root / f"{window.window_id}.json"
        if target.exists():
            raise ValueError("PPO window is already sealed")
        payload = json.dumps(asdict(window), indent=2, sort_keys=True).encode() + b"\n"
        digest = hashlib.sha256(payload).hexdigest()
        with tempfile.NamedTemporaryFile(dir=self.root, delete=False) as temporary:
            temp_path = Path(temporary.name)
            temporary.write(payload)
            temporary.flush()
            os.fsync(temporary.fileno())
        try:
            os.replace(temp_path, target)
        finally:
            temp_path.unlink(missing_ok=True)
        return target, digest


__all__ = ["RolloutWindowStore", "SealedRolloutWindow"]
