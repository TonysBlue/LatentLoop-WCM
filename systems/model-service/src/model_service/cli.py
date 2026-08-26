from __future__ import annotations

import argparse
from pathlib import Path

from omegaconf import OmegaConf
from runtime.codec import CodecIdentity
from runtime.codec_worker import CodecWorkerClient
from runtime.config import ProjectConfig, load_config

from model_service.service import ModelService
from model_service.transport.server import UnixModelServer


def _load_service_config(path: str | Path) -> tuple[ProjectConfig, str | None, str | None]:
    """Load a project config plus optional service-level socket/device settings."""
    config_path = Path(path).expanduser().resolve()
    raw = OmegaConf.to_container(OmegaConf.load(config_path), resolve=False)
    if not isinstance(raw, dict) or "config" not in raw:
        return load_config(config_path), None, None
    project_path = Path(str(raw["config"])).expanduser()
    if not project_path.is_absolute():
        project_path = (config_path.parent / project_path).resolve()
    return (
        load_config(project_path),
        str(raw["socket"]) if raw.get("socket") else None,
        str(raw["device"]) if raw.get("device") else None,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="model-service")
    parser.add_argument("command", choices=("serve",))
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint")
    parser.add_argument("--socket")
    parser.add_argument("--device")
    parser.add_argument("--codec-socket", help="Mimi codec worker Unix socket")
    args = parser.parse_args(argv)
    config, service_socket, service_device = _load_service_config(args.config)
    socket_path = args.socket or service_socket
    if not socket_path:
        parser.error("--socket is required unless the service config provides socket")
    device = args.device or service_device or "cpu"
    decoder = None
    if args.codec_socket:
        decoder = CodecWorkerClient(
            Path(args.codec_socket),
            CodecIdentity(
                config.data.codec_id,
                config.data.codec_weight_hash,
                config.data.codec_revision,
                sample_rate=config.data.audio_sample_rate,
                frame_rate=config.data.codec_frame_rate,
                frame_samples=config.data.unit_audio_samples,
                codebooks=config.data.codec_codebooks,
                codebook_size=config.data.codec_codebook_size,
            ),
        )
        decoder.health()
    UnixModelServer(
        ModelService(config, args.checkpoint, device, speech_decoder=decoder), socket_path
    ).serve_forever()
    return 0
