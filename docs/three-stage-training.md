# 统一三阶段训练架构

> 状态：最终目标训练契约
> 日期：2026-08-20
> 关联文档：[实时流多模态 LatentLoop](realtime-multimodal-latent-loop.md) · [Online RL：Online Recurrent PPO 与隔离环境](online-recurrent-ppo-training.md)

## 1. 总体定义

LatentLoop 的正式训练顺序固定为：

```text
Pretrain -> SFT -> Online RL
```

Canary、Pilot、Production 都完整执行相同的三个阶段。它们使用同一模型语义、
同一 `training.py::train`、同一 `recipe.py::run_recipe`、同一环境协议、同一
reward 公式和同一 checkpoint 格式。Online RL 的当前正式算法固定为
Online Recurrent PPO。规模之间只改变数据数量、更新数、PPO 窗口和计算
资源，不改变单 active lifetime 或训练目标。

模型运行时始终只有两个输出头：Speech Head 直接输出 speech mode 与 Mimi codec token；
Unified Action Head 通过一个结构化 ActionFrame schema 输出全部电脑操作。RL 的 Value Head
是训练专用估值组件，不跨 Model Service 边界，也不承担独立 memory loss。
三个阶段共享同一个 Perceiver、Predictor、WorldStateUpdate、Backbone、PredictionAdapter
和 Future Gate；没有 EMA target encoder 或独立 Reasoner。

## 2. 三类独立数据

三个阶段的数据不能混同：

| 阶段 | 数据来源 | 主要用途 |
|---|---|---|
| Pretrain | 广覆盖多模态监督 episode | 学习音频、屏幕、时间、语言、动作和状态转移的通用规律 |
| SFT | 独立的高质量专家轨迹 | 学习可靠交互、语音响应和电脑操作策略 |
| Online RL | 当前 policy 在真实隔离环境生命期时间线中的窗口 | 使用 Online Recurrent PPO，按感知 Reward Event、交互质量、延迟、效率和安全反馈优化策略 |

Pretrain/SFT episode 按 80 ms unit 保存。没有专家 action 的 unit 必须设置
`action_supervision_mask=false`；不得把“无 action 标签”伪造成有监督 `NO_ACTION`。
同理，缺少 speech 标签时必须关闭相应 speech mask。mask 表示监督是否存在，而非
模型在该时刻应该做什么。

## 3. Pretrain

Pretrain 使用 teacher forcing 的结构化 frame negative log-likelihood：

$$
\mathcal{L}_{\mathrm{pretrain}}
= w_{\mathrm{speech}}
\left(\mathcal{L}_{\mathrm{speech\_mode}}
+ \mathcal{L}_{\mathrm{speech\_codec}}\right)
+ w_{\mathrm{action}}\mathcal{L}_{\mathrm{action\_frame}}
+ \mathcal{L}_{\mathrm{JEPA}}
$$

各项只在自己的有效 mask 上归一化。Speech Head、Unified Action Head、Perceiver、
Predictor、PredictionAdapter/Future Gate、Backbone 和 WorldStateUpdate 全部按各自梯度路径
更新。WorldStateUpdate 没有独立 memory target；未来 speech/action loss 通过
`Z_t -> Backbone -> H_t -> heads` 反向监督记忆更新，JEPA source loss 另外通过 Predictor
监督 Perceiver 和 WorldStateUpdate。

## 4. SFT

SFT 与 Pretrain 使用相同的模型 forward，但 JEPA 系数为 0.5；输入是独立、审核过的专家
交互轨迹。SFT 不是“只训练 head”的适配阶段，全模型继续更新。SFT 最终 checkpoint
同时成为 Online RL 的初始 policy 与冻结 reference policy。

Pretrain checkpoint 只能作为 SFT 的父 checkpoint；SFT 数据身份、stage
和父 checkpoint hash 都进入谱系。SFT 不读取 Pretrain 数据作为隐式回退。

## 5. Online RL

Online RL 当前使用 Online Recurrent PPO，运行在单一生命期
ObservationSignal 时间线上。冻结的外部多模态
Judge 只读取 canonical observation bytes，推断单 active goal 的 Reward Event；完成
finalization watermark 后，后台候选策略用时间折扣 GAE、Value Head 和 clipped policy
ratio 更新。旧策略继续服务，候选通过验证后在 unit 边界原子切换。详细数学定义见
`online-recurrent-ppo-training.md`。

每个 PPO epoch 同时计算当前 sealed on-policy window 与 SFT replay episode 的连续观测
JEPA loss，两路系数分别为 0.1。SFT replay 的行为监督系数仍为 0.1；独立 preservation
数据只执行行为 loss ratio 门禁，不参与任何 optimizer 梯度。

### 5.1 单参数 JEPA 时序

在一个 optimizer 参数版本 $\theta_k$ 内：

$$
P_t = \mathrm{Perceiver}(O_t; \theta_k)
$$

$$
P_{t+1} = \mathrm{Perceiver}(O_{t+1}; \theta_k)
$$

$$
\widehat{P}_{t+1\mid t} = \mathrm{Predictor}(P_t, Z_t)
$$

$$
\mathcal{L}_{\mathrm{JEPA}}
= \mathrm{distance}\!\left(
\widehat{P}_{t+1\mid t}, \mathrm{stopgrad}(P_{t+1})
\right)
+ \mathrm{variance\_floor}(P_t)
$$

完整梯度累积周期结束后才执行 `optimizer.step()`。模型时间步不等于 optimizer step；
同一 JEPA pair 的 source 和 target 始终使用同一参数版本。$P_{t+1}$ 在下一正常时刻仍作为
可微 source 接受行为和 JEPA 梯度。

监督 chunk 最后一项使用 Perceiver-only lookahead 编码下一 observation 和 audio cache 副本，
不更新 Z/H/KV，丢弃返回 cache。episode 最后一项没有 successor 时不形成 pair。

配置不使用重复的 `training.objective`。`training.stage` 唯一决定三阶段分派；
仅当 `stage=rl` 时，`training.rl.algorithm=online_recurrent_ppo` 决定具体 RL 更新规则。

## 6. 当前数据契约

监督 episode 与在线 rollout 统一使用当前数据契约。项目不维护 schema 编号或历史迁移路径。
监督样本 metadata 至少包含：

```text
stage
dataset_scale
sample_kind
supervision_kind
action_source
task_id
environment_id
environment_version
protocol_version
action_schema_id = structured-action-v1
architecture_id = latentloop-perceiver-jepa-v1
runtime_identity
decoded_controls
receipts
```

在线 rollout 还必须记录 lifetime lineage/session ID、sealed window ID、policy/reference hash、
采样 frame、speech/action joint old/reference log-prob、value、Reward Event identity、环境 receipt、
observation/policy-sample hash chain、seed 和 window disposition。旧 flat-action 数据不属于当前
输入；历史资产已清理，后续只从源轨迹生成当前数据。
每个 trainable sealed window 还记录 `lookahead_unit_index=end_unit+1` 与对应 observation
payload SHA-256；该 observation 只用于最后一个 source 的 JEPA target。

## 7. 当前 checkpoint 与阶段谱系

checkpoint 不写入 format/schema 编号，metadata 至少包含：

```text
stage
algorithm
data_identity
codec identity
action_schema_id
architecture_id
parent_sha256
reference_checkpoint_sha256
environment_id
session_manifest_sha256
reward_spec_id
judge_model_id/revision
rubric_sha256
lineage_id
policy_version
observation_chain_sha256
policy_sample_chain_sha256
```

Pretrain checkpoint 的 parent 可以为空；SFT 的 parent 必须是 Pretrain；Online RL 的 parent
必须沿训练更新链前进，同时 reference hash 始终指向冻结的 SFT checkpoint。旧 flat-action
checkpoint 以及旧 InputEncoder/KV/head 布局 checkpoint 直接拒绝，不能 resume 或
warm-start 任意权重。
Pretrain/SFT checkpoint 的 `algorithm=null`；Online RL checkpoint 的
`algorithm=online_recurrent_ppo`。checkpoint 不保存 `objective` 字段。

## 8. 三种规模配置

| 参数 | Canary | Pilot | Production |
|---|---:|---:|---:|
| Pretrain updates | 1,000 | 50,000 | 100,000 |
| SFT updates | 600 | 30,000 | 60,000 |
| Online RL updates | 400 | 20,000 | 40,000 |
| PPO window | 750 units | 750 units | 750 units |
| active lifetime | 1 | 1 | 1 |
| memory / rollout horizon | 750 units | 750 units | 750 units |

Canary 是完整训练链的小规模证明，不是删减版算法。它同样使用真实隔离电脑环境和
连续生命期窗口；只减少数据量、窗口数和 update 数。

## 9. 测试契约

实现必须由以下测试保护：

- 配置拒绝错误 stage/algorithm、正式 RL 缺失环境/Judge identity/socket、非法 PPO 参数；
- 三个正式 recipe 都严格包含 `pretrain -> sft -> rl`，且全部 `backbone_train_mode=all`；
- 当前数据契约往返保存结构化 frame、runtime identity、decoded controls、receipts 与 metadata；
- speech-only 导入保持 action mask 全 false，显式专家动作可正确编码；
- 环境客户端校验 identity，并保证 observation 不携带 reward/隐藏状态；
- 单一 lifetime 时间线不分叉、不从同一初始状态生成 group，并记录 old/reference log-prob；
- time-discount GAE、PPO clipping、Value、KL、窗口封存和全模型梯度可验证；
- JEPA target detach、同参数版本、方差下界、chunk lookahead 不改变 recurrent state 可验证；
- PPO on-policy/replay JEPA 分别加权记录，lookahead 不进入 reward、ratio 或 PPO unit count；
- 行为 loss 在 Predictor 前 stop-gradient，但能训练 Adapter/Gate；JEPA loss 能训练 Predictor、
  source Perceiver 和 WorldStateUpdate；
- 后台 candidate 期间旧 policy 持续服务，stale 窗口隔离，有限性/KL/SFT 保真门禁拒绝时
  serving policy 不变；
- resume 校验完整谱系和当前模型状态，并拒绝不完整 checkpoint；
- Canary、Pilot、Production 通过同一 recipe 和 train 分派路径。

除上述单项契约外，`configs/recipes/smoke.yaml` 必须通过公共
`scripts/run-training.sh -> training run-recipe -> recipe.py::run_recipe ->
training.py::train` 路径真实执行一次 `Pretrain -> SFT -> Online RL`。该 recipe 仅使用
显式 `dataset=synthetic` 数据，并在测试进程中连接实现相同 socket/物理信号协议的
test-only Harness、codec 与 Reward Judge；它必须产出三个阶段的 checkpoint、逐阶段
validation、最终 test 和 recipe report，并校验 SFT 父节点为 Pretrain、Online RL 的父节点
及冻结 reference 均为最终 SFT。该测试闭环不是正式 Canary 的环境或数据替代品，正式
recipe 仍必须连接真实数据资产与隔离 QEMU/KVM 环境，任何缺失都 fail closed。

三个阶段的训练结果必须使用一致的运行观测字段：optimizer update、consumed units、elapsed
seconds、units/second、peak allocated/reserved device memory 和 tracking 实际模式。Pretrain/SFT
另外报告 speech/action supervision density；Online RL 报告 reward、finalization lag、sealed/dropped
window、candidate acceptance 与 reference/preservation gate。所有 lag 指标必须为非负值。
