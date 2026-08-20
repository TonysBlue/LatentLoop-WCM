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
    audit_pilot_data,
    build_pilot_manifest,
    build_pilot_text,
    check_readiness,
    fetch_pilot_data,
    prepare_pilot_data,
    select_pilot_voices,
    synthesize_pilot,
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
    prepare = subparsers.add_parser("prepare-pilot-data")
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
    prepare.add_argument(
        "--dataset", choices=("canary", "pilot", "production", "all"), default="canary"
    )
    prepare.add_argument("--fixture", action="store_true")
    for name in (
        "fetch-pilot-data", "select-pilot-voices", "build-pilot-text",
        "synthesize-pilot", "build-pilot-manifest", "audit-pilot-data",
    ):
        command = subparsers.add_parser(name)
        command.add_argument("--config", required=True)
        command.add_argument("--root")
        command.add_argument("--fixture", action="store_true")
    subparsers.choices["fetch-pilot-data"].add_argument("--lock")
    subparsers.choices["fetch-pilot-data"].add_argument("--download", action="store_true")
    subparsers.choices["fetch-pilot-data"].add_argument("--extract", action="store_true")
    subparsers.choices["select-pilot-voices"].add_argument("--library")
    for name in (
        "build-pilot-text", "synthesize-pilot", "build-pilot-manifest", "audit-pilot-data"
    ):
        subparsers.choices[name].add_argument(
            "--dataset", choices=("canary", "pilot"), required=True
        )
    subparsers.choices["build-pilot-text"].add_argument("--seed", type=int, default=17)
    subparsers.choices["synthesize-pilot"].add_argument("--synth-command")
    subparsers.choices["synthesize-pilot"].add_argument("--asr-command")
    subparsers.choices["synthesize-pilot"].add_argument("--model-sha256")
    subparsers.choices["build-pilot-manifest"].add_argument("--normalize-command")
    subparsers.choices["build-pilot-manifest"].add_argument("--screen-command")
    subparsers.choices["audit-pilot-data"].add_argument("--mimi-report")
    args = parser.parse_args(argv)
    if args.command == "check-readiness":
        config = load_config(args.config)
        root = args.root or config.runtime.data_root
        print(json.dumps(check_readiness(root, config=config), indent=2))
    elif args.command == "prepare-pilot-data":
        config = load_config(args.config)
        root = args.root or config.runtime.data_root
        report = prepare_pilot_data(
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
            dataset=args.dataset,
        )
        print(json.dumps(report, indent=2, default=str))
    elif args.command in {
        "fetch-pilot-data", "select-pilot-voices", "build-pilot-text", "synthesize-pilot",
        "build-pilot-manifest", "audit-pilot-data",
    }:
        config = load_config(args.config)
        root = args.root or config.runtime.data_root
        if args.command == "fetch-pilot-data":
            report = fetch_pilot_data(
                root, fixture=args.fixture, lock_path=args.lock,
                download=args.download, extract=args.extract,
            )
        elif args.command == "select-pilot-voices":
            report = select_pilot_voices(root, library=args.library, fixture=args.fixture)
        elif args.command == "build-pilot-text":
            report = build_pilot_text(
                root, dataset=args.dataset, fixture=args.fixture, seed=args.seed
            )
        elif args.command == "synthesize-pilot":
            report = synthesize_pilot(
                root, dataset=args.dataset, fixture=args.fixture,
                synth_command=args.synth_command, asr_command=args.asr_command,
                model_sha256=args.model_sha256,
            )
        elif args.command == "build-pilot-manifest":
            report = build_pilot_manifest(
                root, dataset=args.dataset, fixture=args.fixture,
                normalize_command=args.normalize_command, screen_command=args.screen_command,
            )
        else:
            report = audit_pilot_data(
                root, dataset=args.dataset, fixture=args.fixture, mimi_report=args.mimi_report
            )
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
                "predictor_layers": config.model.predictor_layers,
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
