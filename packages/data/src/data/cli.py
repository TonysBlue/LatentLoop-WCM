from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

import torch
from media import benchmark_decoder
from model import StreamingLatentLoop
from runtime.config import load_config

from data.capture import (
    CaptureLedger,
    ExpertCaptureSession,
    action_frame_from_dict,
    speech_signal_from_base64,
)
from data.codec_targets import encode_target_speech
from data.curation import (
    audit_canary_data,
    build_canary_manifest,
    build_canary_text,
    check_readiness,
    fetch_canary_data,
    prepare_canary_data,
    select_canary_voices,
    synthesize_canary,
)
from data.curation.prepare import codec_client
from data.overfit import SpeechOverfitDataset
from data.ray import generate_synthetic_with_ray, write_ray_report
from data.replay import replay_episode, replay_ledger
from data.speech_import import import_speech_manifest
from data.synthetic import SyntheticEpisodeDataset
from data.webdataset import EpisodeShardReader, write_episode_shards


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="data")
    subparsers = parser.add_subparsers(dest="command", required=True)
    config_commands = {
        "generate-data", "build-overfit-data", "validate-data", "encode-speech",
        "import-speech", "benchmark-codec", "inspect-model",
    }
    for name in config_commands:
        command = subparsers.add_parser(name)
        command.add_argument("--config", required=True)
        command.add_argument("--set", action="append", default=[], dest="overrides")
    subparsers.choices["generate-data"].add_argument("--output")
    subparsers.choices["generate-data"].add_argument("--ray", action="store_true")
    subparsers.choices["build-overfit-data"].add_argument("--output", required=True)
    subparsers.choices["validate-data"].add_argument("--shards")
    subparsers.choices["encode-speech"].add_argument("--shards")
    subparsers.choices["encode-speech"].add_argument("--output", required=True)
    subparsers.choices["encode-speech"].add_argument("--socket", required=True)
    subparsers.choices["import-speech"].add_argument("--manifest", required=True)
    subparsers.choices["import-speech"].add_argument("--output", required=True)
    subparsers.choices["benchmark-codec"].add_argument("--socket", required=True)
    subparsers.choices["benchmark-codec"].add_argument("--frames", type=int, default=250)
    subparsers.choices["benchmark-codec"].add_argument("--report")
    subparsers.choices["inspect-model"].add_argument("--report")
    capture = subparsers.add_parser("capture")
    capture.add_argument("--config", required=True)
    capture.add_argument("--harness-socket", required=True)
    capture.add_argument("--snapshot", required=True)
    capture.add_argument("--session-id", required=True)
    capture.add_argument("--lineage-id", required=True)
    capture.add_argument("--task-id", required=True)
    capture.add_argument("--split", choices=("train", "validation", "test"), default="train")
    capture.add_argument("--ledger", required=True)
    capture.add_argument("--output", required=True, help="staging shard pattern")
    capture.add_argument("--seed", type=int, default=17)
    capture.add_argument("--input", default="-", help="JSONL expert events or - for stdin")
    capture.add_argument("--viewer-socket")
    replay = subparsers.add_parser("replay")
    replay.add_argument("--harness-socket", required=True)
    replay_source = replay.add_mutually_exclusive_group(required=True)
    replay_source.add_argument("--ledger")
    replay_source.add_argument("--shards")
    replay.add_argument("--config")
    replay.add_argument("--episode-id")
    replay.add_argument("--snapshot", required=True)
    replay.add_argument("--session-id", required=True)
    replay.add_argument("--seed", type=int, default=17)
    replay.add_argument("--non-realtime", action="store_true")
    replay.add_argument("--viewer-socket")
    readiness = subparsers.add_parser("check-readiness")
    readiness.add_argument("--config", required=True)
    readiness.add_argument("--root")
    prepare = subparsers.add_parser("prepare-canary-data")
    prepare.add_argument("--config", required=True)
    prepare.add_argument("--root")
    prepare.add_argument("--lock")
    prepare.add_argument("--download", action="store_true")
    prepare.add_argument("--extract", action="store_true")
    prepare.add_argument("--library")
    prepare.add_argument("--synth-command")
    prepare.add_argument("--asr-command")
    prepare.add_argument("--model-sha256")
    prepare.add_argument("--normalize-command")
    prepare.add_argument("--screen-command")
    prepare.add_argument("--socket")
    prepare.add_argument("--encode", action="store_true")
    prepare.add_argument("--mimi-report-dir")
    prepare.add_argument("--fixture", action="store_true")
    for name in (
        "fetch-canary-data", "select-canary-voices", "build-canary-text",
        "synthesize-canary", "build-canary-manifest", "audit-canary-data",
    ):
        command = subparsers.add_parser(name)
        command.add_argument("--config", required=True)
        command.add_argument("--root")
        command.add_argument("--fixture", action="store_true")
    subparsers.choices["fetch-canary-data"].add_argument("--lock")
    subparsers.choices["fetch-canary-data"].add_argument("--download", action="store_true")
    subparsers.choices["fetch-canary-data"].add_argument("--extract", action="store_true")
    subparsers.choices["select-canary-voices"].add_argument("--library")
    subparsers.choices["build-canary-text"].add_argument("--seed", type=int, default=17)
    subparsers.choices["synthesize-canary"].add_argument("--synth-command")
    subparsers.choices["synthesize-canary"].add_argument("--asr-command")
    subparsers.choices["synthesize-canary"].add_argument("--model-sha256")
    subparsers.choices["build-canary-manifest"].add_argument("--normalize-command")
    subparsers.choices["build-canary-manifest"].add_argument("--screen-command")
    subparsers.choices["audit-canary-data"].add_argument("--mimi-report")
    args = parser.parse_args(argv)
    if args.command == "capture":
        from harness.transport.control import HarnessControlClient

        config = load_config(args.config)
        client = HarnessControlClient(args.harness_socket)
        harness_identity = client.identity()
        ledger = CaptureLedger(
            args.ledger, session_id=args.session_id, lineage_id=args.lineage_id
        )
        session = ExpertCaptureSession(
            client, ledger, seed=args.seed, snapshot_id=args.snapshot
        )
        source = sys.stdin
        viewer: subprocess.Popen[bytes] | None = None
        try:
            session.start()
            viewer = _start_viewer(args.viewer_socket)
            source = (
                sys.stdin
                if args.input == "-"
                else Path(args.input).expanduser().open(encoding="utf-8")
            )
            for line in source:
                if not line.strip():
                    continue
                event = json.loads(line)
                frame = action_frame_from_dict(event.get("action_frame", event))
                speech = speech_signal_from_base64(
                    str(event.get("speech_pcm_b64", "")),
                    silent=bool(event.get("speech_silent", False)),
                )
                session.step(frame, speech)
            episode = session.finish(
                {
                    "codec_id": config.data.codec_id,
                    "codec_weight_hash": config.data.codec_weight_hash,
                    "codec_revision": config.data.codec_revision,
                    **harness_identity,
                    "source": "expert-capture",
                    "source_license": "internal-consented-capture",
                    "redistribution_allowed": False,
                    "split": args.split,
                    "task_id": args.task_id,
                    "session_id_hash": hashlib.sha256(
                        args.session_id.encode()
                    ).hexdigest(),
                    "device_id_hash": hashlib.sha256(
                        harness_identity["environment_id"].encode()
                    ).hexdigest(),
                }
            )
            write_episode_shards([episode], args.output)
        finally:
            if args.input != "-":
                source.close()
            session.close()
            if viewer is not None:
                viewer.terminate()
        print(
            json.dumps(
                {"session_id": args.session_id, "units": len(ledger.records), "sealed": True}
            )
        )
    elif args.command == "replay":
        from harness.transport.control import HarnessControlClient

        client = HarnessControlClient(args.harness_socket)
        viewer: subprocess.Popen[bytes] | None = None

        def start_viewer() -> None:
            nonlocal viewer
            viewer = _start_viewer(args.viewer_socket)

        def print_replay_event(event) -> None:
            print(
                json.dumps(
                    {
                        "unit_index": event.unit_index,
                        "receipt_accepted": event.receipt_accepted,
                        "execution_latency_ms": event.execution_latency_ms,
                        "observation_unit_index": event.observation_unit_index,
                        "elapsed_ms": event.elapsed_ms,
                    },
                    separators=(",", ":"),
                ),
                flush=True,
            )

        try:
            replay_args = {
                "harness": client,
                "snapshot_id": args.snapshot,
                "seed": args.seed,
                "session_id": args.session_id,
                "realtime": not args.non_realtime,
                "step": False,
                "on_started": start_viewer,
                "on_event": print_replay_event,
            }
            if args.ledger:
                events = replay_ledger(args.ledger, **replay_args)
            else:
                if not args.config:
                    raise ValueError("--config is required with --shards")
                config = load_config(args.config)
                events = replay_episode(
                    args.shards,
                    data_config=config.data,
                    model_config=config.model,
                    episode_id=args.episode_id,
                    **replay_args,
                )
        finally:
            if viewer is not None:
                viewer.terminate()
        print(
            json.dumps(
                {
                    "session_id": args.session_id,
                    "units": len(events),
                    "realtime": not args.non_realtime,
                }
            )
        )
    elif args.command == "check-readiness":
        config = load_config(args.config)
        root = args.root or config.runtime.data_root
        print(json.dumps(check_readiness(root, config=config), indent=2))
    elif args.command == "prepare-canary-data":
        config = load_config(args.config)
        root = args.root or config.runtime.data_root
        report = prepare_canary_data(
            root,
            config=config,
            fixture=args.fixture,
            lock_path=args.lock,
            download=args.download,
            extract=args.extract,
            library=args.library,
            synth_command=args.synth_command,
            asr_command=args.asr_command,
            model_sha256=args.model_sha256,
            normalize_command=args.normalize_command,
            screen_command=args.screen_command,
            socket_path=args.socket,
            encode=args.encode,
            mimi_report_dir=args.mimi_report_dir,
        )
        print(json.dumps(report, indent=2, default=str))
    elif args.command in {
        "fetch-canary-data", "select-canary-voices", "build-canary-text", "synthesize-canary",
        "build-canary-manifest", "audit-canary-data",
    }:
        config = load_config(args.config)
        root = args.root or config.runtime.data_root
        if args.command == "fetch-canary-data":
            report = fetch_canary_data(
                root, fixture=args.fixture, lock_path=args.lock,
                download=args.download, extract=args.extract,
            )
        elif args.command == "select-canary-voices":
            report = select_canary_voices(root, library=args.library, fixture=args.fixture)
        elif args.command == "build-canary-text":
            report = build_canary_text(root, fixture=args.fixture, seed=args.seed)
        elif args.command == "synthesize-canary":
            report = synthesize_canary(
                root, fixture=args.fixture,
                synth_command=args.synth_command, asr_command=args.asr_command,
                model_sha256=args.model_sha256,
            )
        elif args.command == "build-canary-manifest":
            report = build_canary_manifest(
                root, fixture=args.fixture,
                normalize_command=args.normalize_command, screen_command=args.screen_command,
            )
        else:
            report = audit_canary_data(root, fixture=args.fixture, mimi_report=args.mimi_report)
        print(json.dumps(report, indent=2, default=str))
    else:
        config = load_config(args.config, args.overrides)
        if args.command == "inspect-model":
            model = StreamingLatentLoop(config.model)
            report = {
                "parameters": model.parameter_count(),
                "tokens_per_unit": config.model.tokens_per_unit,
                "perceiver_slots": config.model.perceiver_slots,
                "perceiver_layers": config.model.perceiver_layers,
                "jepa_layers": config.model.jepa_layers,
                "max_kv_tokens": config.model.kv_units * config.model.perceiver_slots,
            }
        elif args.command == "generate-data":
            output = args.output or str(config.runtime.data_path() / "generated" / "train-%06d.tar")
            manifest = (
                generate_synthetic_with_ray(config, output)
                if args.ray
                else write_episode_shards(
                    SyntheticEpisodeDataset(config.data, config.model), output
                )
            )
            if args.ray:
                write_ray_report(
                    config.runtime.data_path() / "generated" / "ray-report.json", manifest
                )
            report = {"episodes": len(manifest), "output": output}
        elif args.command == "build-overfit-data":
            manifest = write_episode_shards(
                SpeechOverfitDataset(config.data, config.model), args.output
            )
            report = {"episodes": len(manifest), "output": args.output}
        elif args.command == "validate-data":
            source = args.shards or config.data.shards
            if not source:
                raise ValueError("provide --shards or configure data.shards")
            episodes = units = 0
            for episode in EpisodeShardReader(source, config.data, config.model):
                episodes += 1
                units += len(episode.units)
            report = {"episodes": episodes, "units": units}
        elif args.command == "import-speech":
            manifest = write_episode_shards(
                import_speech_manifest(args.manifest, config.data, config.model), args.output
            )
            report = {"episodes": len(manifest), "output": args.output}
        elif args.command == "encode-speech":
            source = args.shards or config.data.shards
            if not source:
                raise ValueError("provide --shards or configure data.shards")
            client = codec_client(config, args.socket)
            client.health()
            manifest = write_episode_shards(
                encode_target_speech(
                    EpisodeShardReader(
                        source,
                        config.data,
                        config.model,
                        require_encoded_speech=False,
                        validate_manifest=False,
                    ),
                    client,
                ),
                args.output,
            )
            report = {"episodes": len(manifest), "output": args.output}
        elif args.command == "benchmark-codec":
            client = codec_client(config, args.socket)
            health = client.health()
            generator = torch.Generator().manual_seed(config.data.seed)
            codes = torch.randint(
                config.data.codec_codebook_size,
                (args.frames, config.data.codec_codebooks, 1),
                generator=generator,
            )
            result = benchmark_decoder(client, codes)
            report = {"health": health, "benchmark": asdict(result)}
            if args.report:
                Path(args.report).expanduser().write_text(
                    json.dumps(report, indent=2) + "\n", encoding="utf-8"
                )
        else:
            raise ValueError(f"unknown data command: {args.command}")
        print(json.dumps(report, indent=2, default=str))
    return 0


def _start_viewer(socket_path: str | None) -> subprocess.Popen[bytes] | None:
    if not socket_path:
        return None
    path = Path(socket_path).expanduser().resolve()
    return subprocess.Popen(
        ["remote-viewer", f"spice+unix://{path}"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
