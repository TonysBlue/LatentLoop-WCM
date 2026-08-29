# 真实 Canary 运行手册

本流程在单张 8 GB NVIDIA GPU 上依次运行 CosyVoice2、Mimi 和主模型，避免三个模型
同时占用显存。数据是固定版本的 AISHELL-1、LibriSpeech、AISHELL-4 和 DailyTalk；
所有 archive、许可证、TTS、ASR 和 Mimi 权重都通过 SHA-256 或固定 revision 校验。
流程不使用 fixture，也不依赖人工审核。

## 前置条件

- WSL2/Linux、NVIDIA GPU 和可用的 `nvidia-smi`
- `uv`、Git、curl，以及至少 20 GB 可用磁盘
- 可访问 GitHub 和 Hugging Face
- 可选：W&B Local 已在 `http://127.0.0.1:8080` 运行并完成 CLI 登录
- Canary Online RL（Online Recurrent PPO）必须连接真实隔离电脑环境 control socket
  `~/latentloop-data/runtime/canary/harness-control.sock` 和 Reward Judge socket
  `~/latentloop-data/runtime/canary/reward-judge.sock`，并准备包含唯一 lifetime
  `session_id/initial_snapshot_id/seed` 的 session manifest；
  连接失败不会回退到 fixture 或离线 rollout。

默认产物位于：

```text
~/latentloop-data/assets/sources/  固定公开数据下载缓存
~/latentloop-data/assets/models/   CosyVoice2、SenseVoice、Mimi 权重
~/latentloop-data/assets/vendor/   固定 revision 的 CosyVoice 源码
~/latentloop-data/datasets/        当前 Canary 数据集
~/latentloop-data/experiments/     checkpoint、评测、日志和 W&B run
~/latentloop-data/runtime/         worker socket 与服务日志
```

## 一键执行

首次运行可将三个 stage 的 update 数覆盖为 1，验证完整 Pretrain → SFT → Online RL 闭环。
离线 tracking 不要求 W&B 登录，但 Online RL（Online Recurrent PPO）仍必须连接真实隔离环境：

```bash
cd ~/LatentLoop
scripts/prepare-data.sh canary all
scripts/run-training.sh --recipe configs/recipes/canary.yaml --run-id canary-001 \
  --set training.max_updates=1 --set training.checkpoint_every=1 \
  --set tracking.mode=offline
```

每个数据准备步骤都有内容哈希和收据，可以安全重跑。使用同一 `--run-id` 重试会复用已验证的
checkpoint；更换 run ID 会创建隔离实验。中断后执行同一命令会复用已验证的下载、规范化音频、
TTS 缓存和 episode。脚本会按阶段启动和关闭 worker，失败时也会清理当前
worker。

训练 recipe 会为每个 stage 保存独立 checkpoint、validation 和最终 test 报告。完整日志保存到本次实验目录：

```text
~/latentloop-data/experiments/canary/<run-id>/<pretrain|sft|rl>/
  checkpoints/
  reports/
  recipe-report.json
```

长阶段每 30 秒打印一次 elapsed time。阶段失败时，终端会自动打印对应日志的最后 60 行；
无需重新运行即可定位失败。需要固定日志路径时设置 `CANARY_LOG_DIR=/path/to/logs`。

## 数据准备与训练检查

需要定位问题或观察产物时使用：

```bash
# 0a. 在本机 GPU 上验证与 Canary 完全同形的 375-unit 更新和 checkpoint
uv run training train --config configs/canary-gpu-smoke.yaml

# 0. 先通过测试专用物理协议实现验证公共三阶段 recipe、谱系和评估闭环
uv run pytest -q tests/integration/test_three_stage_smoke_recipe.py

# 1. 下载并校验真实语料、CosyVoice2 和 SenseVoice
scripts/prepare-data.sh canary bootstrap

# 2. 规范化、生成计划、CosyVoice2 合成、SenseVoice CER/WER 门禁、构造 source manifest
scripts/prepare-data.sh canary prepare

# 3. 从 source manifest 构建当前 staging shards，Mimi decode 门禁、正式审计和三个 split 的 codec 编码
scripts/prepare-data.sh canary encode

# 4. 运行完整三阶段的最小闭环并评测 validation/test
scripts/run-training.sh --recipe configs/recipes/canary.yaml --run-id canary-001 \
  --set training.max_updates=1 --set training.checkpoint_every=1 \
  --set tracking.mode=offline
```

W&B Local 已登录时可把最后一条的 tracking 切成在线：

```bash
scripts/run-training.sh --recipe configs/recipes/canary.yaml --run-id canary-001
```

## 正式 Canary 训练

8 GiB 本机 GPU 在 Online RL candidate 训练期间由 Training System 独占。启动 Mimi worker
时设置 `MIMI_DEVICE=cpu`（或指向另一张 GPU），不得让 codec 与 375-unit PPO candidate
争抢同一张卡：

```bash
MIMI_DEVICE=cpu scripts/codec-worker.sh
```

短闭环成功后运行配置中的完整预算：Pretrain 1,000、SFT 600、Online RL（Online Recurrent PPO）400 updates：

```bash
scripts/run-training.sh --recipe configs/recipes/canary.yaml --run-id canary-001
```

Pretrain 从随机初始化开始；SFT 自动使用 Pretrain checkpoint；Online RL 自动使用最终
SFT checkpoint 作为 policy 和冻结 reference。

三个正式 stage 使用同一条连续 episode 路径：按时间顺序处理每个 episode，持续传递
RecentKV、C、SlowMemory、H、speech-local 和 action-local state。Canary 配置的
`tbptt_units=375` 与
`memory_horizon_units=375`，确保未来输出 loss 可以回传到长期记忆更新器。正式 GPU smoke
保持相同模型形状和 horizon；只有快速 CPU fixture 缩小这些数值，且不作为容量证据。
W&B 中记录 speech mode、有效 codec 帧和 action frame 的监督密度。
训练只使用 speech mode/codec 与 structured action frame loss；长期记忆没有独立
target 或正则项。cosine 学习率最低保持为初始值的 10%。

## 验收产物

成功的一键短闭环至少包含：

```text
~/latentloop-data/datasets/canary/reports/{audit,codec-benchmark,readiness}.json
~/latentloop-data/datasets/canary/shards/processed/{train,validation,test}/*.tar
~/latentloop-data/experiments/canary/<run-id>/pretrain/checkpoints/step-00000001.pt
~/latentloop-data/experiments/canary/<run-id>/sft/checkpoints/step-00000001.pt
~/latentloop-data/experiments/canary/<run-id>/rl/checkpoints/step-00000001.pt
~/latentloop-data/experiments/canary/<run-id>/rl/reports/validation.json
~/latentloop-data/experiments/canary/<run-id>/rl/reports/test.json
~/latentloop-data/experiments/canary/<run-id>/recipe-report.json
```

`canary-audit.json` 必须为 `passed: true`，readiness 必须没有失败项，validation/test
评测必须读取到非零 episode 和 speech frame。`training.json` 记录训练耗时、吞吐、显存、
最后一次训练指标、W&B 实际模式和 run URL。最小闭环只证明真实数据、codec、训练、
checkpoint 和评测链路可运行；此时 codec accuracy、macro-F1 等质量指标不用于判断收敛。

本阶段只验收 Canary。必须提供锁定的真实 source manifest、Reward Judge identity/rubric、
session manifest 和 Mimi worker；缺少任一外部资产时命令以非零状态退出。完成本机报告前
不定义下一规模，报告至少覆盖各阶段耗时、吞吐、consumed units、监督密度、峰值显存和评测计数。

## 自定义目录

所有运行目录都可以通过环境变量覆盖：

```bash
LATENTLOOP_STORAGE_ROOT=/data/latentloop \
LATENTLOOP_DATA_ROOT=/data/latentloop/datasets \
LATENTLOOP_ASSET_ROOT=/data/latentloop/assets \
LATENTLOOP_EXPERIMENT_ROOT=/data/latentloop/experiments \
LATENTLOOP_RUNTIME_ROOT=/data/latentloop/runtime \
scripts/prepare-data.sh canary all
scripts/run-training.sh --recipe configs/recipes/canary.yaml --run-id canary-001
```
