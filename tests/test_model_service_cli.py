from __future__ import annotations

from pathlib import Path

from model_service.cli import _load_service_config


def test_service_config_resolves_project_config_socket_and_device() -> None:
    config, socket, device = _load_service_config(
        Path("configs/services/model/canary.yaml")
    )
    assert config.data.dataset == "canary"
    assert socket == "~/latentloop-data/runtime/canary/model-service.sock"
    assert device == "cuda:0"


def test_project_config_keeps_cli_defaults_unset(tmp_path: Path) -> None:
    path = tmp_path / "project.yaml"
    path.write_text(Path("configs/smoke.yaml").read_text(encoding="utf-8"), encoding="utf-8")
    config, socket, device = _load_service_config(path)
    assert config.data.dataset == "synthetic"
    assert socket is None
    assert device is None
