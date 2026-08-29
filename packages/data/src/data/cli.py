from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import torch
from media import benchmark_decoder
from model import StreamingLatentLoop
from runtime.config import load_config

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
    if args.command == "check-readiness":
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
