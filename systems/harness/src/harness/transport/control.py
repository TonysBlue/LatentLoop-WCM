"""Harness control-plane Unix socket.

The control plane carries operation metadata and base64 protobuf payloads. The
physical data plane remains the Model Service protobuf socket; the Harness
control socket starts one lifetime session and applies physical actuation
without exposing raw model tokens or reward fields to the environment.
"""

from __future__ import annotations

import base64
import binascii
import json
import socket
import threading
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from contracts import (
    ActuationSignal,
    EnvironmentReceipt,
    ObservationSignal,
)
from contracts.framing import frame, read_frame
from contracts.protocol import (
    actuation_to_payload,
    message_to_actuation,
    message_to_observation,
    observation_to_payload,
    payload_to_receipt,
    receipt_to_payload,
)

from harness.action.safety import SafetyGate


def _read_exact(connection: socket.socket, size: int) -> bytes:
    output = bytearray()
    while len(output) < size:
        block = connection.recv(size - len(output))
        if not block:
            raise ConnectionError("Harness control client disconnected")
        output.extend(block)
    return bytes(output)


@dataclass(slots=True)
class _SessionState:
    backend: Any
    initial_snapshot_id: str
    next_unit: int
    observation: ObservationSignal


class HarnessControlServer:
    """Serve one request per Unix socket connection with fail-closed errors."""

    def __init__(
        self,
        backend_factory: Callable[[], Any],
        socket_path: str,
        *,
        expected_environment_id: str | None = None,
        expected_environment_version: str | None = None,
        expected_protocol_version: str = "realtime-v2",
        expected_action_schema_id: str = "structured-action-v1",
        safety_gate: SafetyGate | None = None,
    ) -> None:
        self.backend_factory = backend_factory
        self.socket_path = Path(socket_path).expanduser()
        self._server: socket.socket | None = None
        self.expected_environment_id = expected_environment_id
        self.expected_environment_version = expected_environment_version
        self.expected_protocol_version = expected_protocol_version
        self.expected_action_schema_id = expected_action_schema_id
        self.safety_gate = safety_gate or SafetyGate()
        self._backends: dict[str, _SessionState] = {}
        self._lock = threading.RLock()
        self._closed = threading.Event()

    def serve_forever(self) -> None:
        self.socket_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            self.socket_path.unlink()
        except FileNotFoundError:
            pass
        server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self._server = server
        server.bind(str(self.socket_path))
        server.listen(64)
        try:
            while not self._closed.is_set():
                try:
                    connection, _ = server.accept()
                except OSError:
                    if self._closed.is_set():
                        break
                    raise
                threading.Thread(
                    target=self._serve_connection, args=(connection,), daemon=True
                ).start()
        finally:
            self.close()

    def close(self) -> None:
        self._closed.set()
        server, self._server = self._server, None
        if server is not None:
            server.close()
        with self._lock:
            states, self._backends = self._backends, {}
        for state in states.values():
            try:
                state.backend.close()
            except Exception:
                pass
        try:
            self.socket_path.unlink()
        except FileNotFoundError:
            pass

    def _serve_connection(self, connection: socket.socket) -> None:
        with connection:
            try:
                request = json.loads(read_frame(lambda size: _read_exact(connection, size)))
                response = self._handle(request)
                connection.sendall(frame(json.dumps(response, separators=(",", ":")).encode()))
            except (
                ConnectionError,
                KeyError,
                TypeError,
                ValueError,
                RuntimeError,
                OSError,
                binascii.Error,
            ) as error:
                response = {"ok": False, "error": str(error)}
                connection.sendall(frame(json.dumps(response, separators=(",", ":")).encode()))

    def _handle(self, request: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(request, dict):
            raise TypeError("Harness control request must be an object")
        operation = request.get("operation")
        if operation == "identity":
            backend = self.backend_factory()
            try:
                identity = {
                    "environment_id": str(backend.environment_id),
                    "environment_version": str(backend.environment_version),
                    "protocol_version": self.expected_protocol_version,
                    "action_schema_id": self.expected_action_schema_id,
                }
                if (
                    self.expected_environment_id
                    and identity["environment_id"] != self.expected_environment_id
                ):
                    raise ValueError("Harness environment identity does not match configuration")
                if (
                    self.expected_environment_version
                    and identity["environment_version"] != self.expected_environment_version
                ):
                    raise ValueError("Harness environment version does not match configuration")
                return {"ok": True, "identity": identity}
            finally:
                if hasattr(backend, "close"):
                    backend.close()
        session_id = str(request.get("session_id", ""))
        if not session_id:
            raise ValueError("session_id is required")
        with self._lock:
            if operation == "start_lifetime_session":
                snapshot_id = str(request.get("initial_snapshot_id", ""))
                if not snapshot_id:
                    raise ValueError("initial_snapshot_id is required")
                backend = self.backend_factory()
                if session_id in self._backends:
                    raise ValueError("lifetime session already exists")
                try:
                    observation = backend.start_lifetime_session(
                        snapshot_id, int(request["seed"]), session_id
                    )
                    self._validate_observation(observation, session_id, 0)
                except Exception:
                    if hasattr(backend, "close"):
                        backend.close()
                    raise
                self._backends[session_id] = _SessionState(
                    backend=backend,
                    initial_snapshot_id=snapshot_id,
                    next_unit=0,
                    observation=observation,
                )
                return {
                    "ok": True,
                    "observation": base64.b64encode(observation_to_payload(observation)).decode(),
                }
            if operation == "resume_lifetime_session":
                state = self._backends.get(session_id)
                if state is None:
                    raise KeyError("unknown Harness session")
                expected_next_unit = int(request["expected_next_unit"])
                if expected_next_unit != state.next_unit:
                    raise ValueError("Harness lifetime resume cursor does not match")
                return {
                    "ok": True,
                    "observation": base64.b64encode(
                        observation_to_payload(state.observation)
                    ).decode(),
                }
            if operation == "apply":
                state = self._backends.get(session_id)
                if state is None:
                    raise KeyError("unknown Harness session")
                payload = base64.b64decode(str(request["actuation"]), validate=True)
                output = message_to_actuation(payload)
                if output.session_id != session_id or output.unit_index != state.next_unit:
                    raise ValueError("actuation session or unit is out of order")
                for control in output.controls:
                    self.safety_gate.validate(control)
                observation, receipt = self._apply_backend(state, output)
                self._validate_observation(observation, session_id, state.next_unit + 1)
                if receipt.session_id != session_id or receipt.unit_index != state.next_unit:
                    raise ValueError("backend receipt identity is invalid")
                state.next_unit += 1
                state.observation = observation
                return {
                    "ok": True,
                    "observation": base64.b64encode(observation_to_payload(observation)).decode(),
                    "receipt": base64.b64encode(receipt_to_payload(receipt)).decode(),
                }
            if operation == "close":
                state = self._backends.pop(session_id, None)
                if state is not None:
                    state.backend.close()
                return {"ok": True}
            raise ValueError(f"unknown Harness control operation: {operation}")

    @staticmethod
    def _validate_observation(
        observation: ObservationSignal,
        session_id: str,
        expected_unit: int,
    ) -> None:
        if observation.session_id != session_id or observation.unit_index != expected_unit:
            raise ValueError("backend observation identity is invalid")

    @staticmethod
    def _apply_backend(
        state: _SessionState, output: ActuationSignal
    ) -> tuple[ObservationSignal, EnvironmentReceipt]:
        return state.backend.apply(output)


class HarnessControlClient:
    def __init__(self, socket_path: str, timeout: float = 30.0) -> None:
        self.socket_path = str(Path(socket_path).expanduser())
        self.timeout = timeout

    def _request(self, request: dict[str, Any]) -> dict[str, Any]:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as connection:
            connection.settimeout(self.timeout)
            connection.connect(self.socket_path)
            connection.sendall(frame(json.dumps(request, separators=(",", ":")).encode()))
            response = json.loads(read_frame(lambda size: _read_exact(connection, size)))
        if not response.get("ok", False):
            raise RuntimeError(str(response.get("error", "Harness request failed")))
        return response

    def identity(self) -> dict[str, str]:
        identity = self._request({"operation": "identity"})["identity"]
        required = {
            "environment_id",
            "environment_version",
            "protocol_version",
            "action_schema_id",
        }
        if not required.issubset(identity):
            raise RuntimeError("Harness identity is incomplete")
        return identity

    def start_lifetime_session(
        self, initial_snapshot_id: str, seed: int, session_id: str
    ) -> ObservationSignal:
        if not initial_snapshot_id or not session_id:
            raise ValueError("initial_snapshot_id and session_id are required")
        response = self._request(
            {
                "operation": "start_lifetime_session",
                "initial_snapshot_id": initial_snapshot_id,
                "seed": seed,
                "session_id": session_id,
            }
        )
        observation = message_to_observation(
            base64.b64decode(response["observation"], validate=True)
        )
        if observation.session_id != session_id or observation.unit_index != 0:
            raise RuntimeError("Harness lifetime start returned an invalid observation identity")
        return observation

    def apply(self, output: ActuationSignal) -> tuple[ObservationSignal, EnvironmentReceipt]:
        response = self._request(
            {
                "operation": "apply",
                "session_id": output.session_id,
                "actuation": base64.b64encode(actuation_to_payload(output)).decode(),
            }
        )
        observation = message_to_observation(
            base64.b64decode(response["observation"], validate=True)
        )
        receipt = payload_to_receipt(base64.b64decode(response["receipt"], validate=True))
        if (
            observation.session_id != output.session_id
            or observation.unit_index != output.unit_index + 1
        ):
            raise RuntimeError("Harness apply returned an invalid observation identity")
        if receipt.session_id != output.session_id or receipt.unit_index != output.unit_index:
            raise RuntimeError("Harness apply returned an invalid receipt identity")
        return observation, receipt

    def resume_lifetime_session(
        self, session_id: str, expected_next_unit: int
    ) -> ObservationSignal:
        if not session_id or expected_next_unit < 0:
            raise ValueError("session_id and non-negative lifetime cursor are required")
        response = self._request(
            {
                "operation": "resume_lifetime_session",
                "session_id": session_id,
                "expected_next_unit": expected_next_unit,
            }
        )
        observation = message_to_observation(
            base64.b64decode(response["observation"], validate=True)
        )
        if (
            observation.session_id != session_id
            or observation.unit_index != expected_next_unit
        ):
            raise RuntimeError("Harness lifetime resume returned an invalid observation")
        return observation

    def close(self, session_id: str) -> None:
        try:
            self._request({"operation": "close", "session_id": session_id})
        except RuntimeError:
            pass
