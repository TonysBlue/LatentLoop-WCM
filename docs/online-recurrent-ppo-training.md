# Online RL：Online Recurrent PPO 与真实隔离电脑环境

> 状态：最终目标 Online RL 阶段、Online Recurrent PPO 算法与环境协议
> 日期：2026-08-20
> 关联文档：[统一三阶段训练架构](three-stage-training.md) · [统一电脑动作输出协议](unified-action.md) · [物理 Rollout 闭环](protocols/physical-rollout.md)

## 1. 环境选择

Canary、Pilot、Production 全部使用同一种真实隔离电脑环境。正式 Online RL
当前唯一允许的算法是 Online Recurrent PPO。它使用一个生命期
session 的连续物理时间线；任务完成不会重置模型或环境。只有显式 session close、设备
更换或故障恢复才建立新的 lineage。Canary 不使用 fixture 或离线 rollout 替代环境。

进程内 deterministic environment 只用于单元测试协议和梯度，不允许被正式配置选择，
也不能在真实环境连接失败时自动回退。

## 2. 环境协议

Harness control client 提供以下语义：

```text
identity() -> EnvironmentIdentity
start_lifetime_session(session_id, initial_snapshot) -> ObservationSignal
apply(session_id, unit_index, ActuationSignal)
    -> ObservationSignal + EnvironmentReceipt
close_lifetime_session(session_id)
```

`EnvironmentIdentity` 包含 environment ID、version、protocol version 和 action schema
identity。连接、reset 和每次 apply 都必须校验 session/task/unit 顺序。

模型可见的 `ObservationSignal` 仅包含：

```text
timestamp_ms
delta_ms
mixed_microphone[1920]
screen[3,224,224]
```

每个 unit 都携带一帧完整屏幕；采集缺帧表示为黑帧，不使用 valid 或 revision 分支。

是否继续 rollout 由 Harness receipt 的 `terminated` 控制字段表示；它不属于模型物理
输入，也不携带任务成功原因。

`terminated` 只表示 episode 是否结束，不透露成功原因。task success、评分器内部状态、
权限判断和隐藏 UI 元数据只能进入训练侧 receipt/reward 记录，不能拼进下一 unit 输入。

## 3. 连续时间线与 Reward Event

训练时间线只保存 canonical `ObservationSignal` protobuf bytes 及 hash chain。冻结的
Reward Judge 只能读取这份字节流，不能读取 action、receipt、DOM、隐藏 task 字段或
evaluator 状态。Tracker 同时只维护一个 active goal；目标完成、失败、放弃或被新目标
替代时产生版本化 Reward Event。`uncertain`/`provisional` 事件不产生训练奖励。

```text
ObservationSignal timeline
    -> single active goal tracker
    -> provisional/finalized RewardEvent
    -> reward finalization watermark
    -> sealed PPO window
```

Reward 仍以可审计多分量形式保存：

$$
R_{\mathrm{interaction}}=0.4R_{\mathrm{speech}}+0.3R_{\mathrm{latency}}+0.3R_{\mathrm{efficiency}}
$$

$$
R_t=R_{\mathrm{task}}+0.2R_{\mathrm{interaction}}+R_{\mathrm{safety}}
$$

所有分量都必须由相同的 ObservationSignal 推断；安全执行器是硬约束，不允许 Judge
正分抵消危险操作。
正式 rubric 固定为 `configs/reward/perceptual-v1.yaml`，其内容 SHA-256 与冻结 Judge 的
model ID/revision 一起写入配置和 checkpoint；`unconfigured` 或 hash 不匹配时拒绝启动。

## 4. PPO 窗口

当前 serving policy 在固定窗口内连续运行并记录 recurrent state、sampled action、old
log-prob、value 和 policy version。窗口只有在 reward finalization watermark 覆盖可归因
事件且没有 infrastructure failure 时封存。旧 policy 继续服务，候选 policy 后台训练；
候选通过验证后在 80 ms unit 边界原子切换。切换只原子替换模型参数和 optimizer
状态；serving policy 在该边界已经形成的完整 recurrent state（包括 Z/H/KV、audio cache、
Speech/Action local state 和 unit cursor）原样继承给新参数，不能 reset，也不能用有限观察
历史反事实重算。有限历史无法恢复早期时间线累积进 Z/H 的长期信息，并可能破坏尚未释放的
按键或未完成 UTF-8 等执行状态。

后台训练不得阻塞物理时间线。reward 等待和 candidate optimizer 运行期间，serving policy
仍逐 unit 产生输出；这些新增 unit 继续写入 observation hash chain，但窗口必须以
`eligible_for_update=false` 和明确 disposition 封存，不能混入正在训练的 candidate。
candidate 完成后先在当前 unit 的 actuation/observation 交换结束处执行门禁，再切换下一
unit 使用的 policy version。切换前后的 recurrent state identity、shape、device 和 unit cursor
必须保持不变；下一 unit 才记录新的 policy version。

candidate 接纳门禁固定包含：loss、梯度和全部权重有限；更新后 sampled reference KL 不
超过 `candidate_max_reference_kl`；独立于 replay loss 的顺序 SFT 保真集 loss 不超过旧
serving policy 的 `candidate_max_eval_loss_ratio` 倍。任一门禁失败时 serving policy 和
optimizer 均保持原值，训练记录 rejection；连续失败达到配置上限后 fail closed。
一个 candidate 对同一 sealed on-policy window 执行 `ppo_epochs` 次 recurrent forward 与
optimizer step，每次都相对封存的 old log-prob/value 计算 clipped objective；正式值为 4，
因此 clipping 会约束第二次及后续更新，而不是停留在初始 ratio=1 的形式检查。window 在这些
epoch 内不可修改，也不会与 candidate 训练期间产生的 stale unit 混合。
SFT replay 与 candidate preservation gate 使用两份显式、锁定且互不相同的 WebDataset manifest；
前者参与每个 PPO epoch 的梯度，后者只用于更新前后 loss ratio 门禁。正式运行不得从 RL train
manifest 临时抽两条样本，也不得让 preservation 样本参与 optimizer。

每个 trainable window 还捕获执行最后一个 U_t 后已经返回的 O_(end+1)，作为最后一个
source 的 JEPA-only lookahead。sealed metadata 记录其 unit index 和 payload SHA-256。
lookahead 不进入 PPO unit count、reward、advantage、old/reference log-prob 或 ratio；candidate
从 window start state 重放到 O_end 后，仅用其 audio cache 副本执行 Perceiver target 编码，
不推进 Z/H/KV。该 observation 后续仍按正常生命期顺序被 serving policy 消费，不能额外执行
一次环境 action。

每个 PPO epoch 同时计算两路 JEPA：sealed on-policy 连续 observation 学习当前物理交互，
SFT replay 连续 episode 稳定通用表示。两路都使用 candidate 内唯一一份 Perceiver 参数，
分别加权和记录；preservation manifest 只做行为门禁，不参与 JEPA 训练。

## 5. Recurrent PPO 数学目标

使用时间折扣和 GAE：

$$
\begin{aligned}
\gamma_t &= \exp\!\left(-\frac{\Delta t_t}{\tau}\right), \\
\delta_t &= r_t + \gamma_t m_t V_{t+1} - V_t.
\end{aligned}
$$

$$
\begin{aligned}
A_t &= \delta_t + \gamma_t\lambda m_t A_{t+1}, \\
y_t &= A_t + V_t.
\end{aligned}
$$

对 speech unit 和 structured action frame 分别计算 ratio，再等权组合：

$$
\begin{aligned}
\log \pi_{\mathrm{speech},t}
&= \log \pi(m_t\mid s_t) \\
&\quad + \mathbf{1}_{\{m_t=\mathrm{speech}\}}
\sum_q \log \pi(c_{t,q}\mid s_t,m_t,c_{t,<q})
\end{aligned}
$$

Speech mode 与当前 unit 所有有效 codec token 构成一个联合动作，因此这里必须求和，不能按
token 数取平均；否则 PPO ratio 与 reference KL 会随 codebook 数被错误缩放。

$$
\rho_t
= \exp\!\left(
\log\pi_\theta(a_t\mid s_t)
- \log\pi_{\mathrm{old}}(a_t\mid s_t)
\right)
$$

$$
\mathcal{L}_{\mathrm{actor}}
= -\mathbb{E}\!\left[
\min\!\left(
\rho_t A_t,
\operatorname{clip}(\rho_t,1-\epsilon,1+\epsilon)A_t
\right)
\right]
$$

$$
\begin{aligned}
\mathcal{L}
={}& \mathcal{L}_{\mathrm{actor}}
+ c_v\mathcal{L}_{\mathrm{value}}
- c_H\mathcal{H}(\pi) \\
&+ \beta D_{\mathrm{KL}}
\left(\pi_\theta\,\|\,\pi_{\mathrm{SFT}}\right)
+ 0.1\mathcal{L}_{\mathrm{SFT}} \\
&+ 0.1\mathcal{L}_{\mathrm{JEPA,on\text{-}policy}}
+ 0.1\mathcal{L}_{\mathrm{JEPA,replay}}.
\end{aligned}
$$

Value Head 是训练专用组件，不跨 Model Service 物理边界输出。Reward/Judge 输出全部
stop-gradient。PPO 不再采样同一初始状态的 rollout group，也不使用组内标准化 advantage。
sampled KL 的 log-ratio 在进入指数前按协议上限截断，以避免极小概率 token 造成数值溢出；
这项数值边界必须在指标中与 candidate 的未截断身份和配置一起审计。

## 6. 在线性与安全边界

- rollout 必须由当前 policy 在线产生，训练不得从静态文件冒充 on-policy 数据；
- policy update 后未消费的旧 rollout 必须丢弃；
- 每个 action frame 在当前 unit 解码后由 Harness 验证 schema、权限和危险操作；
- 环境超时、崩溃、身份变化或 receipt 不完整时，该 rollout 标为 infrastructure failure，
  不当作低 reward 样本训练；
- 环境 socket 和 snapshot 位于仓库外，连接失败必须 fail closed；
- policy、reference、session manifest、Judge、环境和 reward spec identity 全部进入 checkpoint。

## 7. Checkpoint 与同一生命期恢复

checkpoint 保存 serving policy/optimizer、policy 与冻结 reference 的完整 recurrent state、
下一个 observation unit cursor、policy version、active goal tracker、reward watermark、timeline
chain hash 和冻结 SFT checkpoint 路径/hash。恢复不是重新启动快照：Training System 必须调用
`resume_lifetime_session(session_id, expected_next_unit)` 连接 Harness 中仍存活的同一 session，
并要求 Harness 当前 observation、timeline 长度、两个 recurrent state 的 unit cursor 和
checkpoint chain hash 完全一致。任一对象缺失或错位都拒绝恢复，绝不通过 reset 伪造连续性。
SFT checkpoint 必须逐键覆盖包含 Value Head 的当前完整模型；replay 与 preservation manifest
的内容 hash 也进入 checkpoint，任一数据内容在同一路径被替换都拒绝恢复。

达到正式训练预算后显式 close lifetime session；因 `stop_after_updates` 暂停时只断开训练
client，保留 Harness、环境与 codec session 供同一 checkpoint 恢复。

## 8. 规模参数

三种规模使用同一协议，默认 window 都是 750 units；Canary、Pilot、Production 只改变
窗口数、更新预算和资源规格。当前正式语义是一个智能体、一个 active lifetime session、
一条连续时间线。多环境并行代表多个智能体/生命期，不属于这套单智能体自学习契约，不能
作为吞吐优化暗中引入。

## 9. 测试契约

- 延迟 Reward Event 必须回填到其 `outcome_unit`，watermark 未覆盖时继续服务且超限失败；
- candidate 训练期间至少有旧 policy 单元继续进入 timeline，且 stale window 永不参与更新；
- finite、KL 或 SFT 保真门禁拒绝 candidate 时 serving 权重逐位不变；
- policy 只在 unit 边界切换，并完整继承该边界的 recurrent state，不 reset 或反事实重算；
- 暂停后从同一 Harness session 恢复，next unit、policy/reference state 和 chain hash 连续；
- Harness session 不存在、cursor 错位、timeline 被修改或 reference checkpoint hash 不符时拒绝恢复。
- window lookahead 必须严格为 `end_unit+1` 且 payload hash 匹配；它不得改变 PPO 样本数、
  reward、advantage 或 ratio；
- on-policy/replay JEPA 分别产生有限 loss 和 Predictor 梯度，preservation episode 不得进入
  optimizer；
- 行为/RL 梯度在 predicted slots 后停止但继续训练 Future Adapter/Gate，JEPA target 侧无梯度。
