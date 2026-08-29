# LatentLoop

实时流多模态 LatentLoop 的研究实现。当前基础设施已经包含有界逐层 RecentKV、Backbone
内固定容量语义 slots `C_t`、每层固定容量 SlowMemory、单路混合麦克风、屏幕输入、直接
codec token Speech Head、独立 Action Head、递归重计算、精确断点恢复、WebDataset、
CPU-only Ray 任务和 W&B Local。

完整架构见 `docs/realtime-multimodal-latent-loop.md`，本地训练设计见
`docs/local-training-platform.md`。直接流式语音的实现与运行命令见
`docs/direct-speech.md`；统一电脑操控的 Action Head 协议见
`docs/unified-action.md`。

## 环境初始化

```bash
scripts/bootstrap.sh
uv run data inspect-model --config configs/smoke.yaml
uv run pytest
```

`data inspect-model` 会验证配置并给出模型参数和 KV 规模。三个配置档位的参数量可用以下
命令确认；GPU FP16 前反向由测试和正式训练启动门禁验证：

```bash
uv run data inspect-model --config configs/smoke.yaml
uv run data inspect-model --config configs/local-dev.yaml
uv run data inspect-model --config configs/research-0.2b.yaml
```

当前档位参数量约为 0.0003B、0.0507B 和 0.2424B。0.0507B 与 0.2424B 档位均已在
RTX 2080 SUPER 8GB 上完成 FP16 反向和 AdamW 更新；实际长期训练仍应以运行时记录的
`runtime/peak_memory_*` 指标为准。

## 数据与训练

生成并校验合成 WebDataset：

```bash
uv run data generate-data \
  --config configs/smoke.yaml \
  --output "$HOME/latentloop-data/datasets/generated/train-%06d.tar"

uv run data validate-data \
  --config configs/smoke.yaml \
  --shards "$HOME/latentloop-data/datasets/generated/train-*.tar"
```

32 轨迹真实 Mimi 过拟合门禁使用 `configs/direct-speech-overfit.yaml`，完整命令和本机
验收结果见 [`docs/direct-speech.md`](docs/direct-speech.md)。

使用 Ray 的 CPU worker 生成数据：

```bash
uv run data generate-data \
  --config configs/smoke.yaml \
  --output "$HOME/latentloop-data/datasets/generated/train-%06d.tar" \
  --ray
```

运行 smoke 训练或恢复当前 checkpoint：

```bash
uv run training train --config configs/smoke.yaml
uv run training train \
  --config configs/smoke.yaml \
  --resume "$HOME/latentloop-data/checkpoints/smoke/step-00000002.pt"
```

训练数据默认写入 `~/latentloop-data/datasets`；当前实验、checkpoint、W&B 和运行时文件分别
写入 `experiments/`、`checkpoints/`、`tracking/` 和 `runtime/`，不保留历史运行产物。

## W&B Local

```bash
scripts/wandb-local.sh up
scripts/wandb-local.sh status
```

服务只监听 `http://127.0.0.1:8080`。首次启动需要完成数据库迁移；脚本以 `/ready`
为就绪条件。打开 UI 获取本地 API key 后执行：

```bash
uv run wandb login --host http://127.0.0.1:8080
uv run training train \
  --config configs/smoke.yaml \
  --set tracking.enabled=true \
  --set tracking.mode=online
```

服务不可用或未登录时，Tracker 自动写入本地 offline run。服务管理命令：

```bash
scripts/wandb-local.sh logs
scripts/wandb-local.sh down
```
直接流式语音使用 80 ms 主时钟与 Mimi 24 kHz codec。实现和运行命令见
[`docs/direct-speech.md`](docs/direct-speech.md)。

Canary 数据准备、外部语料许可证锁、CosyVoice/ASR/屏幕适配器以及自动审计门禁见
[`docs/canary-data.md`](docs/canary-data.md)。本地 fixture 可以在不下载语料和模型的
情况下跑通准备链：

真实一小时 Canary 的固定数据下载、TTS/ASR、Mimi 编码、训练和 validation/test 评测
使用统一入口，完整说明见 [`docs/canary-runbook.md`](docs/canary-runbook.md)：

```bash
scripts/prepare-data.sh canary all
scripts/run-training.sh --recipe configs/recipes/canary.yaml --run-id canary-001
```

```bash
uv run data fetch-canary-data --config configs/local-dev.yaml --fixture
uv run data select-canary-voices --config configs/local-dev.yaml --fixture
uv run data build-canary-text --config configs/local-dev.yaml --fixture
uv run data synthesize-canary --config configs/local-dev.yaml --fixture
uv run data build-canary-manifest --config configs/local-dev.yaml --fixture
uv run data audit-canary-data --config configs/local-dev.yaml --fixture
```
