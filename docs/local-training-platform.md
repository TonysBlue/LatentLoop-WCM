# LatentLoop 本地训练平台与工程实施方案

> 状态：最终目标工程契约
> 日期：2026-08-20
> 关联顶层架构：[实时流多模态 LatentLoop 完整方案](realtime-multimodal-latent-loop.md)
> 输出协议：[直接流式语音实施说明](direct-speech.md) · [统一电脑动作输出协议](unified-action.md)
> 训练协议：[统一三阶段训练架构](three-stage-training.md) · [Online RL：Online Recurrent PPO 与真实隔离电脑环境](online-recurrent-ppo-training.md)
> 目标：在本地 WSL2/单 GPU 环境中提供与最终目标架构一致、可恢复、可观测的训练和验证闭环。

## 1. 设计结论

本地平台直接实现顶层架构，不维护语义不同的过渡 head 或阶段专用训练循环：

$$
P_t = \mathrm{Perceiver}(O_t)
$$

$$
Z_t = \mathrm{WorldStateUpdate}(Z_{t-1}, H_{t-1})
$$

$$
\widehat{P}_{t+1\mid t} = \mathrm{Predictor}(P_t, Z_t)
$$

$$
F_t = \mathrm{PredictionAdapter}\!\left(\mathrm{stopgrad}(\widehat{P}_{t+1\mid t})\right) + E_{\mathrm{future}}
$$

$$
(H_t, \mathrm{KV}_t) = \mathrm{Backbone}
\left(P_t, Z_t, F_t, \mathrm{KV}_{t-1}\right)
$$

$H_t$ 随后同时进入 Speech Head 和 Unified Action Head；训练使用 masked loss 与
TBPTT，并把结果写入带 lineage 的原子 checkpoint。

PyTorch 负责模型、递归状态、loss、优化和 checkpoint；WebDataset 负责按 episode 顺序提供 unit；Ray 负责 CPU 数据与环境外围任务；W&B Local 负责指标和谱系。KV、Z、H、audio cache 和两个 local state 保留在同一 GPU 进程。

## 2. 当前环境与约束

### 2.1 硬件基线

| 项目 | 目标环境 |
|---|---|
| 操作系统 | Windows 11 + WSL2 Ubuntu |
| GPU | RTX 2080 SUPER 8 GiB |
| CUDA 精度 | Turing 使用 FP16，不依赖 BF16 |
| Attention | PyTorch SDPA 或普通 attention |
| GPU worker | 单训练/推理进程 |
| Ray | CPU only |
| W&B Local | 127.0.0.1 本地服务 |

显存上限以专用显存为准，不依赖 Windows 共享 GPU 内存。训练、codec、checkpoint 和数据文件放在 WSL Linux 文件系统。

### 2.2 运行边界

- 训练直接运行在 WSL2，不把 GPU 核心循环放入 Docker。
- W&B Local 是观测服务，不进入模型 forward。
- Ray Actor 必须声明 `num_gpus=0`。
- 模型、递归状态、codec worker 和 Harness 接口均使用版本化 identity。
- 任何异常必须显式失败、隔离或恢复，不能静默改变训练语义。

### 2.3 环境门槛

开始训练前验证：

```text
nvidia-smi 可识别目标 GPU
torch.cuda.is_available() == True
FP16 forward/backward/optimizer.step 成功
峰值显存可统计
checkpoint 可原子写入 Linux 文件系统
codec worker health 和 identity 校验通过
```

## 3. 工程目录

仓库保存代码、配置、文档、测试和索引；大文件位于仓库外：

```text
LatentLoop/
├── configs/
├── docs/
├── packages/
│   ├── contracts/     # 物理信号与控制面 protobuf/类型
│   ├── model/         # Model Core：Z/H/KV、Speech Head、Action Head
│   ├── media/         # PCM 与媒体时钟校验
│   ├── data/          # episode、WebDataset、curation、readiness
│   └── runtime/       # 配置、codec worker、action continuation
├── systems/
│   ├── model-service/ # ObservationSignal -> ActuationSignal 推理边界
│   ├── training/      # Pretrain/SFT/Online RL（Online Recurrent PPO）、checkpoint、评估
│   └── harness/       # QEMU/KVM、采集、执行、receipt、感知 Judge client
├── scripts/
└── tests/

~/latentloop-data/
├── assets/
├── datasets/
├── experiments/
├── checkpoints/
├── runtime/
├── tracking/
```

数据、音频、codec 权重、socket、checkpoint 和 W&B 媒体不提交 Git。manifest、hash、resolved config 和 parent checkpoint 组成实验谱系。

## 4. 依赖与复现

### 4.1 Python 环境

使用仓库锁定的 Python/uv 环境和项目 lockfile。核心依赖包括 PyTorch、Accelerate、OmegaConf、WebDataset、safetensors、W&B SDK、Ray 和开发测试工具。依赖升级必须验证 CPU 测试、FP16 前反向、checkpoint 恢复、WebDataset 往返和 codec identity。

### 4.2 配置优先级

配置按以下顺序合并：

```text
dataclass schema -> YAML profile -> CLI --set
```

每次 run 保存完整 resolved config，而不是只保存 CLI 差异。配置必须包含模型形状、codec、unit 时钟、latent/KV、TBPTT、loss 权重、数据 manifest、随机种子、checkpoint 和 tracking。

### 4.3 统一代码路径

Canary、Smoke 和 direct-speech gate 使用同一个 train/recipe/evaluate 代码路径。Canary 是当前唯一正式规模，按 `Pretrain -> SFT -> Online RL` 运行；Online RL 采用 Online Recurrent PPO。Smoke 和 gate 只用于本机测试，不构成新的规模。`training.py::train` 直接按 `stage` 分派：Pretrain/SFT 使用共享监督分支，Online RL 使用 `training.rl.algorithm` 指定的在线环境分支。后续规模基于 Canary 实测结果另行设计，不预留复制的 Python、shell 或 YAML 路径。

## 5. 多模态时间契约

### 5.1 统一时间轴

生产 unit 为 80 ms：

```text
unit_ms            = 80
audio_sample_rate  = 24000
audio_samples      = 1920
codec_frame_rate   = 12.5
codec_frames/unit  = 1
```

### 5.2 StreamUnit

```python
@dataclass
class StreamUnit:
    timestamp_ms: Tensor
    delta_ms: Tensor
    mic_audio: Tensor
    screen: Tensor
    speech_mode: Tensor
    speech_mode_mask: Tensor
    speech_codes: Tensor
    speech_codec_mask: Tensor
    action: ActionFrame
    action_supervision_mask: Tensor
```

`screen` 始终为当前 224x224 RGB 像素；采集缺帧时由边界生成黑帧。静态与动态画面
使用同一视觉编码路径。`delta_ms` 为正，时间戳严格递增；所有 target 都有对应 mask。

### 5.3 编码和状态顺序

$$
P_t = \mathrm{Perceiver}(O_t)
$$

$$
Z_t = \mathrm{WorldStateUpdate}(Z_{t-1}, H_{t-1})
$$

$$
\widehat{P}_{t+1\mid t} = \mathrm{Predictor}(P_t, Z_t)
$$

$$
F_t = \mathrm{PredictionAdapter}\!\left(\mathrm{stopgrad}(\widehat{P}_{t+1\mid t})\right) + E_{\mathrm{future}}
$$

$$
(H_t, \mathrm{KV}_t) = \mathrm{Backbone}
\left(P_t, Z_t, F_t, \mathrm{KV}_{t-1}\right)
$$

$$
U_t = \left(
\mathrm{SpeechHead}(H_t, \mathrm{speech\_local}_{t-1}),
\mathrm{ActionHead}(H_t, \mathrm{action\_local}_{t-1})
\right)
$$

`P_t` 和 `H_t` 都保存16个 slots，H_t 不能被 pooled summary 替代。audio cache 是
Perceiver 流式输入状态，不进入顶层认知状态公式。

### 5.4 Codec 时间对齐

每个 unit 固定一帧 Mimi code。音频、完整屏幕帧、speech mask 和 action mask 都按照同一 unit 时钟对齐，不允许累计四舍五入。

## 6. 公共训练接口

### 6.1 RecurrentState

```python
@dataclass
class RecurrentState:
    layer_kv: tuple[LayerKV, ...]
    latent: Tensor
    audio_cache: Tensor
    hidden: Tensor
    speech_local: SpeechLocalState
    action_local: ActionLocalState
    unit_index: Tensor
```

`hidden` 保存完整上一 unit 的 H；`detach()` 对所有递归浮点张量截断计算图，但不 reset 数值状态。

### 6.2 StepOutput

```python
@dataclass
class StepOutput:
    state: RecurrentState
    speech_mode_logits: Tensor
    speech_codec_logits: Tensor
    action: ActionHeadOutput
    hidden: Tensor
    perceiver_slots: Tensor
    predicted_next_slots: Tensor
    future_gate_mean: Tensor
    future_gate_max: Tensor
```

forward 不执行操作系统动作、W&B 网络调用或全局 session 读取。

### 6.3 神经 codec 接口

codec identity 必须包含：

```text
codec_id = mimi-24khz-8x2048
sample_rate = 24000
frame_rate = 12.5
frame_samples = 1920
codebooks = 8
codebook_size = 2048
revision
weight_sha256
```

数据、配置、runtime 和 checkpoint 必须使用同一 identity。

## 7. 模型结构

### 7.1 数据流

```text
MIC_MIXED + SCREEN + TIME
        -> Perceiver(P_t)
Z_(t-1) + H_(t-1)
        -> WorldStateUpdate -> Z_t
P_t + Z_t
        -> Predictor -> P_hat_(t+1|t)
stop_grad(P_hat)
        -> PredictionAdapter -> F_t
P_t + Z_t + F_t + KV_(t-1)
        -> Streaming Backbone -> H_t, KV_t
H_t -> Speech Head + Unified Action Head
```

### 7.2 Streaming Backbone

每层执行 cached self-attention、周期性 World cross-attention、门控 Future cross-attention、
feed-forward 和 final normalization。同一 unit 的16个 slots双向互见，只读取历史 unit KV。
KV 不区分视觉和非视觉，统一保留最近 `kv_units * 16` 个当前 slots；Z/F 不作为额外 token
直接写入 KV。

所有 profile 固定 `perceiver_slots=16`、`perceiver_layers=2`、`predictor_layers=2`；
Smoke 只缩小 model_dim、Backbone layers、KV horizon 和数据预算，不改变 JEPA 模块拓扑。
Perceiver 以16个 learned queries 两次执行 input cross-attention、slot self-attention 和 FFN。
Predictor 以 P_t 为固定顺序 slots，通过 Z cross-attention 预测一个80 ms后的同形表示。

Future Adapter 使用 `LayerNorm -> identity-initialized Linear`。每个 Future 层根据当前 hidden
和 cross-attention context 生成 `[B,16,1]` gate；gate projection 零初始化且 bias 为 -4。
行为梯度在 P_hat 后停止，因此训练 Adapter/Gate 而不进入 Predictor。

### 7.3 WorldStateUpdate

$$
Z_t = \mathcal{U}_{\theta}\left(Z_{t-1}, H_{t-1}\right)
$$

内部使用 latent projection、对 H 的 slot cross-attention、learned slot identity、candidate 和
per-slot/per-dimension gate。WorldStateUpdate 不接收 `delta_t`、当前 audio/vision 或动作与语音输出；
它也没有 memory target、probe、write 或 diversity loss。`delta_t` 只经过 DeltaTimeEncoder 后作为
当前 Backbone 的输入特征。

`Z_t` 是抽象 latent workspace，可以在一次离散更新中重组、压缩或跳转，不要求满足严格的物理
时间动力学。系统只维护一个 `latent` 字段，不区分预测状态和观测校正状态。

### 7.4 DeltaTimeEncoder

`timestamp_ms` 不作为模型语义输入。只有当前观察间隔 `delta_ms` 经过 DeltaTimeEncoder 后
进入 Backbone：

$$
T_t^{\Delta} = \mathrm{DeltaTimeEncoder}(\Delta t_t)
$$

$\mathrm{WorldStateUpdate}(Z_{t-1}, H_{t-1})$ 的接口不包含 $\Delta t_t$。

### 7.4 Speech Head

每 80 ms 预测 mode 和一帧 Mimi codec。SILENCE unit 的 codec mask 为 false；speech local 只负责声学连续性。

### 7.5 Unified Action Head

所有 action kind 和 kind-conditioned 参数共用 Structured ActionFrame schema、context、
local state 和 frame joint probability。每 unit frame 立即解码执行；只有 TYPE decoder 的
UTF-8 pending bytes 跨 unit。Harness 负责 schema、安全和权限校验。

## 8. 模型配置档位

### 8.1 Smoke

Smoke 只缩小 model_dim、layers、screen shape、KV horizon 和数据量，用于张量、梯度、数据和恢复测试。它仍使用 80 ms unit、相同状态转移和相同 loss 代码。

三阶段 Smoke 使用 `configs/recipes/smoke.yaml` 和 `configs/stages/smoke-*.yaml`，通过公共
recipe runner 顺序执行 Pretrain、SFT 与 Online Recurrent PPO。监督阶段使用显式 synthetic
episode；RL 阶段只允许测试进程提供的 test-only Harness、codec 和 Reward Judge socket，
并继续传输规范的 `ObservationSignal`/`ActuationSignal`。该配置用于验证训练分派、在线窗口、
checkpoint 谱系和 evaluation/report 闭环，不得作为 Canary 的环境回退。

### 8.2 Local

Local profile 用于单 GPU 完整结构验证，包含音频/视觉 encoder、latent updater、750-unit 生产形状可配置的 KV、两个 output heads、checkpoint 和 W&B。

### 8.3 Canary 正式契约

Canary profile 使用正式 codec、trajectory/action schema、60 秒 KV（750 units）和 memory
horizon 750。短回归可以覆盖 update 数和 tracking mode，但不得改变状态和数据协议。模型宽度、
数据规模或硬件预算的扩展必须等待本机 Canary 报告并单独更新设计。

## 9. 数据格式

### 9.1 WebDataset episode

每个 episode 主要保存：

```text
meta.json
mic.flac
target_speech.flac
screen.npz
timeline.npz
speech_codes.npy
turns.json
```

训练协议不读取旧 `controls.npy` 或 memory target；历史资产已清理，后续只从源轨迹构建当前数据。

### 9.2 Split 规则

按完整 device/session 分组切分 train/validation/test；同一 session 不跨 split。合成场景按模板、seed 和环境条件隔离。

### 9.3 数据输入边界

`mic.flac` 是模型唯一音频输入；`target_speech.flac` 只用于离线 codec 编码、审计和评测。TTS、来源分离和环境参数只能用于数据准备，不能进入模型输入。

### 9.4 数据校验

训练前验证时间戳、音频长度、codec frame 对齐、屏幕帧形状、action grammar、mask、manifest/content hash、license 和 split leakage。损坏 episode 进入 quarantine，不静默跳过。

## 10. 训练循环

### 10.1 单卡执行

```text
state = model.initial_state()
for unit in episode chronological order:
    output = model.forward_step(unit, state, teacher_targets)
    losses = compute_losses(output, unit)
    accumulate(losses)
    state = output.state
backward at TBPTT boundary
optimizer.step()
state = state.detach()
```

episode 边界 reset；TBPTT 边界 detach。不能每个 unit reset 或把递归 state 移出训练进程。

### 10.2 优化配置

optimizer、学习率、梯度累积、FP16、梯度裁剪和 checkpoint cadence 由配置定义。Canary 的 `tbptt_units` 和 `memory_horizon_units` 为 750；Smoke 可缩小但不能换代码路径。

### 10.3 Loss 契约

$$
\mathcal{L}_{\mathrm{speech}}
= \mathcal{L}_{\mathrm{speech\_mode}}
{}+ \mathcal{L}_{\mathrm{speech\_codec}}
$$

$$
\mathcal{L}_{\mathrm{action}}
= \mathrm{MaskedStructuredActionNLL}
\left(\mathrm{action\_output}, \mathrm{action\_frame}\right)
$$

$$
\mathcal{L}_{\mathrm{JEPA}}
= \mathcal{L}_{\mathrm{normalized\_slot\_prediction}}
{}+ \mathcal{L}_{\mathrm{latent\_variance\_floor}}
$$

$$
\mathcal{L}_{\mathrm{total}}
= w_{\mathrm{speech}}\mathcal{L}_{\mathrm{speech}}
{}+ w_{\mathrm{action}}\mathcal{L}_{\mathrm{action}}
{}+ w_{\mathrm{JEPA}}^{(\mathrm{stage})}\mathcal{L}_{\mathrm{JEPA}}
$$

SILENCE unit 的 codec loss 被 mask。Action 参数按 kind 激活，连续参数 NLL 与 rollout
log-prob 使用同一 bounded 参数化。Z 没有 memory probe、write-budget、diversity、control
或 confidence loss；JEPA 是 Predictor 的单步表示目标，不是 Z 的独立 memory target。

### 10.4 梯度与模块影响

| 模块 | 监督来源 | 主要影响 |
|---|---|---|
| Speech Head | mode/codec loss | speech mode、codec 和局部连续性 |
| Action Head | structured frame NLL | kind、参数、TYPE continuation |
| Perceiver | Speech、Action/RL、JEPA source | 多模态观察表示 |
| Predictor | JEPA | 单步未来表示 |
| Future Adapter/Gate | Speech、Action/RL | 受控未来先验 |
| Backbone | Speech、Action/RL | 共享多模态表示 |
| WorldStateUpdate/Z | 未来行为 loss 与 JEPA source | 长期目标、约束和抽象任务状态 |
| KV | 无参数 loss | 近期精确上下文 |
| speech/action local | 对应 head loss | 跨帧/跨 unit 连续性 |

### 10.5 长时监督

```text
future Speech/Action loss
 -> future H
 -> future Z
 -> earlier WorldStateUpdate
```

TBPTT 不能短于要验证的 memory horizon；生产使用 750 units。

### 10.6 Lookahead 与参数版本

chunk 内 $P_{t+1}$ 同时作为下一时刻正常可微 source 和上一时刻 detach target。chunk 最后
一个 source 若还有 successor，只调用同一 Perceiver 编码 O_(t+1) 和 audio cache 副本；
不更新 Z/H/KV 并丢弃新 cache。完整梯度累积周期使用同一参数版本，只有 sync 边界执行
optimizer step；模型 unit index 不能被当作 optimizer version。

## 11. Checkpoint 与恢复

### 11.1 保存内容

checkpoint 保存：

```text
model weights
optimizer/scheduler/scaler state
Python/NumPy/Torch/CUDA RNG
RecurrentState: Z, H, KV, audio_cache,
                speech_local, action_local, unit_index
epoch/shard/sample/unit cursor
global update and consumed units
resolved config + config hash
data manifest hash
codec identity
parent checkpoint hash
git commit
```

### 11.2 原子写入

临时文件写入同一文件系统，flush/fsync，计算 SHA-256，原子 rename，fsync 目录，再更新 manifest 和 tracking。中断不能破坏最近可用 checkpoint。

### 11.3 恢复校验

恢复时检查 model shape、unit clock、trajectory/action schema、codec identity、manifest
hash、config hash 和 parent checkpoint compatibility。递归状态恢复后下一 unit 的
output/loss 应与连续运行一致。

## 12. W&B Local

W&B 只记录观测和谱系，不保存完整原始数据或替代 checkpoint。run 记录：

- resolved config/config hash；
- git commit 和 dirty 状态；
- data manifest hash；
- codec identity；
- model/parameter count；
- GPU、CUDA、PyTorch；
- seed、split、run/stage identity；
- total、speech、action、latent on/off、KV、queue 和 system metrics。

原始音视频不上报；屏幕媒体必须脱敏。服务不可用时切换 offline，不中断训练。

## 13. Ray 外围编排

Ray 负责 CPU 数据生成、音频处理、屏幕变化检测、shard 校验、离线评测和环境 Actor。Ray 不负责：

- 每个 unit 的核心 train/forward；
- GPU 参数或 recurrent state；
- Object Store 中的 KV、Z、H 或 playback queue；
- 竞争同一 GPU 的多个训练 trial。

## 14. 验收与测试

### 14.1 环境

CUDA、FP16、codec worker、checkpoint 原子写入和 W&B Local/Offline 可用。

### 14.2 数据与协议

StreamUnit shape、80 ms 时钟、mask、manifest/shard identity、runtime identity、codec identity、action schema 和 split isolation 通过。

### 14.3 状态闭环

KV 长度不超过配置上限；淘汰只发生在完整 unit 边界；Z/H/local state 形状固定；chunk 内有跨 unit 梯度，chunk 外图已 detach；latent on/off 的长时行为可测。

### 14.4 语音与 action

speech mode/codebook accuracy、SILENCE codec mask、action kind/参数指标、schema validity、
TYPE 跨 unit continuation、session/unit 顺序和 Harness safety gate 通过。

### 14.5 性能与稳定性

峰值显存、unit latency、p95/p99、queue、socket、NaN、丢帧、长时恢复和 checkpoint consistency 达到配置门槛。

## 15. W&B/运行时故障处理

| 故障 | 行为 |
|---|---|
| CUDA OOM | 保存诊断，从最近完整 checkpoint 恢复 |
| NaN/Inf | 停止异常 step，记录输入、loss、grad norm |
| 数据损坏 | quarantine episode，训练计数不前进 |
| W&B 断开 | 切换 offline，不中断训练 |
| codec/socket 失败 | 显式失败或重启恢复，不改变状态语义 |
| 磁盘不足 | 停止新写入，保护已有 checkpoint 和 manifest |

## 16. 安全与故障边界

- 原始屏幕和麦克风最小权限保存；
- device/session hash 不可逆；
- W&B 媒体脱敏；
- action 风险由 Harness 审批、白名单、速率限制和紧急停止控制；
- 模型 forward 不能绕过 Harness 直接执行操作；
- 删除数据时同步更新 manifest 和谱系。

## 17. 最终工程定义

```text
chronological WebDataset episodes
-> StreamUnit(80 ms mixed mic + screen)
-> Perceiver(O_t) = P_t
-> Z_t = WorldStateUpdate(Z_(t-1), H_(t-1))
-> Predictor(P_t, Z_t) = P_hat_(t+1|t)
-> PredictionAdapter(stop_grad(P_hat_(t+1|t))) + E_future = F_t
-> H_t, KV_t = Backbone(P_t, Z_t, F_t, KV_(t-1))
-> Speech Head + Unified Action Head
-> masked Speech/Action/JEPA loss + TBPTT
-> atomic checkpoint + W&B lineage
```

本地平台的目录、配置、数据、训练、恢复、测试和运行时语义都直接服务于顶层最终架构，不维护与其不一致的中间实现。
