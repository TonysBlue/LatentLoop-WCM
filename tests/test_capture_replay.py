from __future__ import annotations

import base64

import numpy as np
import pytest
from contracts import (
    ActionFrame,
    ActionKind,
    ActuationSignal,
    EnvironmentReceipt,
    MicSignal,
    ObservationSignal,
    ScreenSignal,
    SpeechSignal,
)
from data.capture import CaptureLedger, ExpertCaptureSession
from data.replay import replay_ledger
from data.webdataset import write_episode_shards


class RecordingHarness:
    def __init__(self) -> None:
        self.outputs: list[ActuationSignal] = []
        self.closed: list[str] = []

    def start_lifetime_session(
        self, initial_snapshot_id: str, seed: int, session_id: str
    ) -> ObservationSignal:
        return self._observation(session_id, 0)

    def identity(self) -> dict[str, str]:
        return {
            "environment_id": "test-physical",
            "environment_version": "1",
            "protocol_version": "realtime-v2",
            "action_schema_id": "structured-action-v1",
        }

    def apply(
        self, output: ActuationSignal
    ) -> tuple[ObservationSignal, EnvironmentReceipt]:
        self.outputs.append(output)
        return (
            self._observation(output.session_id, output.unit_index + 1),
            EnvironmentReceipt(output.session_id, output.unit_index, True),
        )

    def close(self, session_id: str) -> None:
        self.closed.append(session_id)

    @staticmethod
    def _observation(session_id: str, unit_index: int) -> ObservationSignal:
        return ObservationSignal(
            session_id,
            unit_index,
            unit_index * 80,
            80,
            MicSignal(np.zeros(1920, dtype=np.float32).tobytes()),
            ScreenSignal(b"x" * (224 * 224 * 3), 224, 224),
        )


def _speech(value: float = 0.0) -> SpeechSignal:
    pcm = np.full(1920, value, dtype=np.float32).tobytes()
    return SpeechSignal(pcm, silent=value == 0.0)


def _codec_metadata() -> dict[str, str]:
    return {
        "codec_id": "mimi-24khz-8x2048",
        "codec_weight_hash": "hash",
        "codec_revision": "revision",
        "environment_id": "test-physical",
        "environment_version": "1",
        "protocol_version": "realtime-v2",
        "action_schema_id": "structured-action-v1",
    }


def test_expert_capture_persists_and_exports_current_episode(tmp_path) -> None:
    harness = RecordingHarness()
    ledger = CaptureLedger(tmp_path / "capture", session_id="capture-1", lineage_id="line-1")
    session = ExpertCaptureSession(harness, ledger, seed=7, snapshot_id="base")
    session.start()
    session.step(ActionFrame(ActionKind.NO_ACTION), _speech(0.25))
    episode = session.finish(_codec_metadata())
    session.close()

    assert (tmp_path / "capture" / "sealed.json").is_file()
    assert (tmp_path / "capture" / "transitions.jsonl").is_file()
    assert len(ledger.records) == len(episode.units) == 1
    assert episode.units[0].action_supervision_mask.item()
    assert episode.units[0].speech_codec_mask.item()
    assert episode.units[0].speech_mode.item() == 1
    assert episode.target_speech is not None
    assert float(episode.target_speech.mean()) == pytest.approx(0.25)
    assert harness.closed == ["capture-1"]


def test_capture_refuses_non_contiguous_results(tmp_path) -> None:
    ledger = CaptureLedger(tmp_path / "capture", session_id="capture-1", lineage_id="line-1")
    ledger.start(RecordingHarness._observation("capture-1", 0))
    output = ActuationSignal("capture-1", 0, _speech())
    with pytest.raises(ValueError, match="next observation identity"):
        ledger.append(
            output,
            RecordingHarness._observation("capture-1", 2),
            EnvironmentReceipt("capture-1", 0, True),
            ActionFrame(ActionKind.NO_ACTION),
        )


def test_model_free_replay_uses_recorded_speech_and_actions(tmp_path, monkeypatch) -> None:
    capture_harness = RecordingHarness()
    ledger = CaptureLedger(tmp_path / "capture", session_id="capture-1", lineage_id="line-1")
    capture = ExpertCaptureSession(capture_harness, ledger, seed=7, snapshot_id="base")
    capture.start()
    capture.step(ActionFrame(ActionKind.NOOP), _speech(0.5))
    capture.finish(_codec_metadata())
    capture.close()

    replay_harness = RecordingHarness()
    monkeypatch.setattr("data.replay.time.sleep", lambda _seconds: None)
    events = replay_ledger(
        tmp_path / "capture",
        harness=replay_harness,
        snapshot_id="base",
        seed=8,
        session_id="replay-1",
        realtime=True,
    )

    assert len(events) == 1
    assert events[0].receipt_accepted
    assert replay_harness.outputs[0].speech.pcm == _speech(0.5).pcm
    assert len(replay_harness.outputs[0].controls) == 1
    assert replay_harness.closed == ["replay-1"]


def test_replay_rejects_invalid_speech_before_harness_apply() -> None:
    from data.replay import replay_capture

    harness = RecordingHarness()
    with pytest.raises(ValueError, match="speech PCM"):
        replay_capture(
            harness,
            [
                {
                    "unit_index": 0,
                    "action_frame": {"kind": "NO_ACTION"},
                    "speech_pcm_b64": base64.b64encode(b"short").decode(),
                }
            ],
            session_id="replay",
            snapshot_id="base",
            seed=1,
            realtime=False,
        )
    assert not harness.outputs


def test_replay_rejects_nonphysical_webdataset_audio_clock(
    tmp_path, smoke_config
) -> None:
    from data import SyntheticEpisodeDataset
    from data.replay import replay_episode

    episode = SyntheticEpisodeDataset(smoke_config.data, smoke_config.model).make_episode(0)
    episode.metadata.update(RecordingHarness().identity())
    write_episode_shards([episode], tmp_path / "train-%06d.tar")
    harness = RecordingHarness()

    with pytest.raises(ValueError, match="physical 1920-sample"):
        replay_episode(
            str(tmp_path / "train-*.tar"),
            data_config=smoke_config.data,
            model_config=smoke_config.model,
            harness=harness,
            snapshot_id="base",
            seed=1,
            session_id="replay",
            episode_id=episode.episode_id,
        )


def test_sealed_capture_rejects_payload_tampering(tmp_path) -> None:
    harness = RecordingHarness()
    ledger = CaptureLedger(tmp_path / "capture", session_id="capture-1", lineage_id="line-1")
    capture = ExpertCaptureSession(harness, ledger, seed=7, snapshot_id="base")
    capture.start()
    capture.step(ActionFrame(ActionKind.NO_ACTION), _speech())
    capture.finish(_codec_metadata())
    capture.close()
    payload = tmp_path / "capture" / "actuation-000000000000.pb"
    payload.write_bytes(payload.read_bytes() + b"tampered")

    with pytest.raises(ValueError, match="payload hash"):
        CaptureLedger.load_sealed(tmp_path / "capture")


def test_replay_rejects_unsealed_capture(tmp_path) -> None:
    harness = RecordingHarness()
    ledger = CaptureLedger(tmp_path / "capture", session_id="capture-1", lineage_id="line-1")
    capture = ExpertCaptureSession(harness, ledger, seed=7, snapshot_id="base")
    capture.start()
    capture.step(ActionFrame(ActionKind.NO_ACTION), _speech())
    capture.close()

    with pytest.raises(ValueError, match="not sealed"):
        replay_ledger(
            tmp_path / "capture",
            harness=RecordingHarness(),
            snapshot_id="base",
            seed=1,
            session_id="replay",
            realtime=False,
        )
