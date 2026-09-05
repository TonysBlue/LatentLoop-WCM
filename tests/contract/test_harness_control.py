from __future__ import annotations

import base64
import threading
import time

import pytest
from contracts import (
    ActuationSignal,
    ButtonPhase,
    ControlKind,
    ControlSignal,
    EnvironmentReceipt,
    MicSignal,
    ObservationSignal,
    ScreenSignal,
    SpeechSignal,
)
from contracts.protocol import actuation_to_payload
from harness.action.safety import SafetyGate
from harness.transport.control import HarnessControlClient, HarnessControlServer


class FakeBackend:
    environment_id = "isolated-qemu-v1"
    environment_version = "1"

    def start_lifetime_session(
        self, initial_snapshot_id: str, seed: int, session_id: str
    ) -> ObservationSignal:
        return ObservationSignal(
            session_id,
            0,
            0,
            80,
            MicSignal(b"x" * 7680),
            ScreenSignal(b"rgb", 1, 1),
        )

    def apply(self, output: ActuationSignal):
        return (
            ObservationSignal(
                output.session_id,
                output.unit_index + 1,
                80,
                80,
                MicSignal(b"x" * 7680),
                ScreenSignal(b"rgb", 1, 1),
            ),
            EnvironmentReceipt(output.session_id, output.unit_index, True),
        )

    def close(self):
        return None


class CountingBackend(FakeBackend):
    def __init__(self, closed: list[str]) -> None:
        self.closed = closed

    def close(self):
        self.closed.append("closed")


def test_harness_control_socket_uses_physical_signals(tmp_path) -> None:
    path = tmp_path / "harness.sock"
    thread = threading.Thread(
        target=HarnessControlServer(lambda: FakeBackend(), str(path)).serve_forever,
        daemon=True,
    )
    thread.start()
    for _ in range(100):
        if path.exists():
            break
        time.sleep(0.01)
    client = HarnessControlClient(str(path))
    assert client.identity()["environment_id"] == "isolated-qemu-v1"
    assert client.start_lifetime_session("snapshot", 7, "session").unit_index == 0
    output = ActuationSignal("session", 0, SpeechSignal(b"y" * 7680, silent=True))
    next_observation, receipt = client.apply(output)
    assert next_observation.unit_index == 1
    assert receipt.accepted
    assert client.resume_lifetime_session("session", 1) == next_observation
    client.close("session")


def test_harness_control_rejects_order_errors_and_duplicate_lifetime_session() -> None:
    closed: list[str] = []
    server = HarnessControlServer(lambda: CountingBackend(closed), "/tmp/unused.sock")
    server._handle(
        {
            "operation": "start_lifetime_session",
            "initial_snapshot_id": "snapshot",
            "seed": 1,
            "session_id": "session",
        }
    )
    with pytest.raises(ValueError, match="already exists"):
        server._handle(
            {
                "operation": "start_lifetime_session",
                "initial_snapshot_id": "snapshot",
                "seed": 2,
                "session_id": "session",
            }
        )
    assert closed == []
    with pytest.raises(ValueError, match="cursor does not match"):
        server._handle(
            {
                "operation": "resume_lifetime_session",
                "session_id": "session",
                "expected_next_unit": 1,
            }
        )
    with pytest.raises(ValueError, match="out of order"):
        server._handle(
            {
                "operation": "apply",
                "session_id": "session",
                "actuation": base64.b64encode(
                    actuation_to_payload(
                        ActuationSignal("session", 4, SpeechSignal(b"y" * 7680, silent=True))
                    )
                ).decode(),
            }
        )
    server.close()
    assert closed == ["closed"]


def test_harness_control_applies_safety_gate_before_backend() -> None:
    server = HarnessControlServer(
        FakeBackend,
        "/tmp/unused.sock",
        safety_gate=SafetyGate(require_approval_for={ControlKind.POINTER_BUTTON}),
    )
    server._handle(
        {
            "operation": "start_lifetime_session",
            "initial_snapshot_id": "snapshot",
            "seed": 1,
            "session_id": "session",
        }
    )
    output = ActuationSignal(
        "session",
        0,
        SpeechSignal(b"x" * 7680, silent=True),
        controls=(
            ControlSignal(
                ControlKind.POINTER_BUTTON,
                "click",
                button=0,
                button_phase=ButtonPhase.CLICK,
            ),
        ),
    )
    with pytest.raises(PermissionError, match="requires approval"):
        server._handle(
            {
                "operation": "apply",
                "session_id": "session",
                "actuation": base64.b64encode(actuation_to_payload(output)).decode(),
            }
        )
    server.close()
