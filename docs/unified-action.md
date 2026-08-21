# 统一电脑动作输出协议

> 状态：最终目标 Structured ActionFrame 协议
> 日期：2026-08-20
> 关联顶层架构：[实时流多模态 LatentLoop 完整方案](realtime-multimodal-latent-loop.md)
> 对称语音协议：[直接流式语音实施说明](direct-speech.md)

## 1. 完成边界

Unified Action Head 是模型唯一的电脑操控输出头。统一的是 Action 的语义 schema、
80 ms 执行边界和联合概率接口，不要求 kind 与不同类型的参数共享扁平 token 序列。
每个 unit 都产生一个完整的 `ActionFrame`，Model Service 在该 unit 内把 frame 解码成
零个或多个有序 `ControlSignal`，Harness 校验后立即执行。

$$
\begin{aligned}
P_t &= \mathrm{Perceiver}(O_t), \\
Z_t &= \mathrm{WorldStateUpdate}(Z_{t-1}, H_{t-1}), \\
\widehat{P}_{t+1\mid t} &= \mathrm{Predictor}(P_t, Z_t), \\
F_t &= \mathrm{PredictionAdapter}\!\left(
\mathrm{stopgrad}(\widehat{P}_{t+1\mid t})
\right) + E_{\mathrm{future}}, \\
(H_t, \mathrm{KV}_t) &= \mathrm{Backbone}
\left(P_t, Z_t, F_t, \mathrm{KV}_{t-1}\right), \\
\mathrm{frame}_t
&= \mathrm{ActionHead}(H_t, \mathrm{action\_local}_{t-1}), \\
\mathrm{controls}_t &= \mathrm{decode}(\mathrm{frame}_t).
\end{aligned}
$$

Action Head 不调用操作系统。Model Service 和 Harness 之间只传递物理
`ControlSignal`，不传递模型 logits、训练 target 或 action 参数 token。

## 2. ActionFrame schema

### 2.1 Kind

`kind` 是七分类变量，协议顺序固定为：

```text
NO_ACTION
NOOP
POINTER_MOVE
POINTER_BUTTON
SCROLL
TYPE
HOTKEY
```

- `NO_ACTION` 表示本 unit 不提交任何 `ControlSignal`；它也是正常的可监督决策；
- `NOOP` 表示显式提交一个无副作用控制事件，供协议/评测需要；
- 等待由连续的 `NO_ACTION` 表达，不存在 `WAIT`；
- 拖拽由 `POINTER_BUTTON(DOWN) -> POINTER_MOVE* -> POINTER_BUTTON(UP)` 组成；
- 双击由两个连续的 `POINTER_BUTTON(CLICK)` 组成；
- 右键由 `POINTER_BUTTON(button=RIGHT, phase=CLICK)` 表达；
- 不存在 `CANCEL`、`DRAG`、`DOUBLE_CLICK`、`RIGHT_CLICK` 宏 kind；
- frame 本身就是 unit 边界，不存在 `END_ACTION` 或 `PAD`。

### 2.2 参数

逻辑 schema 为：

```text
ActionFrame {
    kind
    coordinate_cell
    coordinate_residual
    button
    button_phase
    scroll_delta
    text_bytes
    text_length
    hotkey_keys
    hotkey_length
}
```

只有当前 kind 对应的参数有语义并参与概率和监督：

| kind | 有效参数 |
|---|---|
| `NO_ACTION` / `NOOP` | 无 |
| `POINTER_MOVE` | `coordinate_cell`, `coordinate_residual` |
| `POINTER_BUTTON` | `button`, `button_phase` |
| `SCROLL` | `scroll_delta` |
| `TYPE` | `text_length`, `text_bytes[:text_length]` |
| `HOTKEY` | `hotkey_length`, `hotkey_keys[:hotkey_length]` |

`POINTER_BUTTON` 作用于 Harness 的当前指针位置，不携带坐标。一个 frame 可以解码为
多个有序控制事件；例如 HOTKEY 先按下各键，再按反序释放。

## 3. 参数域与确定性解码

### 3.1 坐标

绝对坐标使用 32x32 joint coarse grid categorical 和 cell 内 bounded residual：

$$
\begin{aligned}
c_x &= \min\!\left(31,
\left\lfloor 32\,\mathrm{clamp}(x,0,1)\right\rfloor\right), \\
c_y &= \min\!\left(31,
\left\lfloor 32\,\mathrm{clamp}(y,0,1)\right\rfloor\right), \\
c &= 32c_y + c_x, \\
(r_x,r_y) &= (32x-c_x,\ 32y-c_y),
\qquad (r_x,r_y)\in[0,1]^2, \\
\widehat{x} &= \frac{c_x+r_x}{32}, \\
\widehat{y} &= \frac{c_y+r_y}{32}.
\end{aligned}
$$

边界值 1.0 映射到最后一个 cell 且 residual 为 1.0。分类项表达全局多峰位置，
bounded residual 提供 cell 内精度。动态画面中的运动与动作时延由连续视觉流训练学习，
执行协议不绑定 screen revision。

### 3.2 Pointer button 与 scroll

button 是 `{LEFT, MIDDLE, RIGHT}` 分类变量，phase 是 `{CLICK, DOWN, UP}` 分类变量。
`DOWN/UP` 更新 action local 的 held-button 状态，Harness 仍需在 session reset 和紧急停止
时释放残留按键。

scroll 是二维连续量 `(dx, dy) in [-1, 1]^2`。Harness 通过版本化适配器把归一化量
转换成物理滚轮 tick 或触控板距离。

### 3.3 TYPE

TYPE 使用纯 UTF-8 byte decoder，每个 unit 最多输出 16 bytes。decoder 可以跨 unit
保留至多 3 个尚未构成完整 Unicode scalar 的 pending bytes：

1. 本 unit byte chunk 追加到 pending bytes；
2. 确定性 UTF-8 assembler 取出所有已经完成且合法的最长前缀；
3. 完成的 `text_chunk` 在当前 unit 立即形成 `TEXT_INPUT` 并执行；
4. 不完整尾部留在 `action_local`，下一 unit 继续；
5. 连续 TYPE 隐式续接；切换到其他 kind 时 pending 必须为空，否则 frame 非法。

continuation 仅存在于 Action Head local state，不出现在公开 `ActionFrame` schema。
已经执行的文本不回滚；模型通过后续真实屏幕和声音观察执行结果，需要纠正时继续输出
退格、快捷键或新的文本动作。

### 3.4 HOTKEY

HOTKEY 使用版本化 32-key table，每 frame 最多 8 keys 且至少一个。key ID 是逻辑键，
平台 scan code 映射由 Harness 管理。解码产生有序的 `KEY_PRESS*` 和反序
`KEY_RELEASE*`，因此不会把“组合键”塞进一个平台相关的宏字段。

## 4. Action Head

Action Head 读取当前 `H_t` 和 action-local state，先预测 kind，再只激活对应参数分支：

$$
\begin{aligned}
C_t^{\mathrm{state}}
&= \mathrm{Attention}(Q^{\mathrm{state}}, H_t, H_t), \\
C_{t,0:16}^{\mathrm{spatial}}
&= \mathrm{Attention}(Q_{0:16}^{\mathrm{spatial}}, H_t, H_t), \\
C_t &= f\!\left(C_t^{\mathrm{state}}, E_{t-1}^{\mathrm{frame}}\right), \\
K_t &\sim \mathrm{Categorical}
\left(\mathrm{KindLogits}(C_t)\right).
\end{aligned}
$$

```text
POINTER_MOVE   -> spatial_context + joint cell categorical + bounded residual distribution
POINTER_BUTTON -> button categorical + phase categorical
SCROLL         -> bounded continuous distribution
TYPE           -> length categorical + autoregressive byte decoder
HOTKEY         -> length categorical + autoregressive key decoder
```

这仍然是一个 Unified Action Head：参数分支由同一个 kind 决策条件化，共享 context、
状态、概率对象和训练/rollout 接口，不是多个可独立调用的动作 head。16 个 learned spatial
queries 为 32x32 coarse cell 分布提供空间读取能力，但不假定 H_t 的 16 个 Perceiver slots
与 4x4 屏幕位置一一对应。视觉编码器不直接连接 Action Head；动作头只读取已经融合音频、
时间、视觉和历史状态的 Backbone hidden。

## 5. Action local state

跨 unit state 保存结构连续性，而不是未执行的宏事件：

```text
ActionLocalState {
    previous_frame_embedding
    type_decoder_state
    pending_utf8_bytes[3]
    pending_utf8_length
    type_active
    held_buttons[3]
    held_keys[32]
}
```

state 在 episode/session 边界 reset，在 TBPTT 边界 detach 但不清空。每个 frame 都能
独立进入执行边界；不存在等待完整 event、原子提交或协议级 rollback。

## 6. 每 unit 执行语义

每 80 ms unit 的顺序固定为：生成 frame、校验当前 kind 参数、解码 controls、Harness
安全校验、按序提交可执行 controls。TYPE 的完整 UTF-8 前缀、POINTER_MOVE、button 和
scroll 都在当前 unit 执行。

执行拒绝、receipt、accepted、safety 和 reward 是 Harness/Training control-plane
metadata，绝不进入下一 `ObservationSignal`。模型只能从下一 unit 的屏幕、混合麦克风
和时间看到动作结果。没有撤销协议；纠错本身也是后续动作。

## 7. 数据契约

### 7.1 Unit targets

当前数据契约的每个 unit 保存结构化 target：

```text
action_kind                 [B]
action_supervision_mask     [B]
action_coordinate_cell      [B]
action_coordinate_residual  [B, 2]
action_button               [B]
action_button_phase         [B]
action_scroll_delta         [B, 2]
action_text_bytes           [B, 16]
action_text_length          [B]
action_hotkey_keys          [B, 8]
action_hotkey_length        [B]
```

`action_supervision_mask=false` 表示该 unit 没有 action 标签；这不同于有监督的
`NO_ACTION`。kind-conditioned 参数 mask 由 kind 和 length 确定，不持久化重复 mask。

### 7.2 数据来源与校验

轨迹按 unit 记录动作、当时的完整屏幕帧、后续物理观察和审计 metadata。模型输入
不得包含未来结果、receipt、隐藏 DOM、特权坐标或教师计划。导入时拒绝：

- 不完整轨迹 metadata 或错误 `action_schema_id`；
- kind 与参数域不一致、坐标/scroll 越界、非法 button/phase/key；
- TYPE 超过 16 bytes、无效 UTF-8 状态转移或跨非 TYPE frame 留有 pending bytes；
- HOTKEY 长度不在 1..8；
- 屏幕帧缺失且未按协议填充黑帧，或时间线倒退；
- 任何旧 `action_tokens`/`action_token_mask` flat representation。

## 8. 训练目标与概率接口

一个 frame 的条件化联合 log-prob 为：

$$
\begin{aligned}
\log p(\mathrm{frame}\mid s)
={}& \log p(K\mid s) \\
&+ \mathbf{1}_{\{K=\mathrm{POINTER\_MOVE}\}}
\left[\log p(c\mid s,K)+\log p(r\mid s,K,c)\right] \\
&+ \mathbf{1}_{\{K=\mathrm{POINTER\_BUTTON}\}}
\left[\log p(b\mid s,K)+\log p(\varphi\mid s,K)\right] \\
&+ \mathbf{1}_{\{K=\mathrm{SCROLL}\}}\log p(d\mid s,K) \\
&+ \mathbf{1}_{\{K=\mathrm{TYPE}\}}
\left[\log p(\ell\mid s,K)+\sum_i\log p(x_i\mid x_{<i},s,K,\ell)\right] \\
&+ \mathbf{1}_{\{K=\mathrm{HOTKEY}\}}
\left[\log p(\ell\mid s,K)+\sum_i\log p(k_i\mid k_{<i},s,K,\ell)\right].
\end{aligned}
$$

监督目标是上述有效项的负 log-likelihood。连续参数使用有界分布的 NLL；实现可用
固定尺度的 bounded regression NLL，但必须和 sampling/log-prob 使用同一参数化。
各分支先按有效 frame 归一化，再组成唯一 $\mathcal{L}_{\mathrm{action}}$，避免 TYPE 长度自动
放大其权重。

Online Recurrent PPO 保存并重算同一个 frame joint log-prob。clipped ratio 与 reference KL 以
frame 为动作概率单位，而不是把 kind、每个 byte 和坐标重复当作独立环境 step。

$$
\mathcal{L}_{\mathrm{total}}
= w_{\mathrm{speech}}\mathcal{L}_{\mathrm{speech}}
+ w_{\mathrm{action}}\mathcal{L}_{\mathrm{action}}
+ w_{\mathrm{JEPA}}^{(\mathrm{stage})}\mathcal{L}_{\mathrm{JEPA}}
$$

Action loss 通过 Action Head、Backbone、Perceiver、Future Adapter/Gate 和
WorldStateUpdate 训练行为路径；它在 predicted slots 处 stop-gradient，不训练 Predictor。
Predictor 由 JEPA loss 训练。没有额外 action-control、confidence、success、memory 或
rollback loss。

## 9. 推理与 Harness 安全边界

Harness 在提交每个 `ControlSignal` 前校验 schema identity、参数范围、
held-input 状态、应用/区域白名单、速率和权限。危险操作仍可被审批或拒绝，session reset
与全局紧急停止负责释放 held inputs。模型输出不是授权，安全拒绝也不改变模型物理输入
边界。

## 10. Checkpoint 与恢复

Checkpoint 保存 Action Head 参数、完整 action local state、
`action_schema_id=structured-action-v1`、grid/type/hotkey/key-table 常量以及 unit/session
identity。恢复后下一 frame 的 logits、采样和 UTF-8 assembler 状态必须与不中断运行一致。

不完整 checkpoint、flat-token 数据和旧配置字段直接拒绝；没有 vocabulary 映射、形状碰巧一致时
的部分加载或运行时兼容 decoder。历史资产已清理，后续只从源轨迹生成当前数据。

## 11. 验证设计

### 11.1 Schema 与 decode

- 七种 kind 的 frame 构造、校验与 ControlSignal 往返；
- 32x32 joint cell 的边界、cell 内 residual 和像素还原；
- button CLICK/DOWN/UP、拖拽序列、双击序列和 HOTKEY press/release 顺序；
- TYPE 中英文和 emoji 分块、跨 unit pending bytes、切 kind 时非法尾部拒绝；
- 每 unit 立即产生可执行 controls，且不需要 END_ACTION。

### 11.2 Head、loss 与状态

- teacher forcing 和 sampling 的全部 tensor shape；
- kind-conditioned mask 只监督当前参数分支；
- action frame joint log-prob 与监督 NLL 使用相同分解；
- action 监督全 false 时 loss 有限且无虚假梯度；
- checkpoint 中断恢复、TBPTT detach 和 session reset；
- Action loss 到达 Action Head、Backbone、Perceiver、Future Adapter/Gate 和
  WorldStateUpdate，但不到达 Predictor；
- learned state/spatial queries 对任意 slot 排列都保持 shape 和概率接口，不依赖旧
  `STATE_QUERY`/`VISION_*` 位置。

### 11.3 边界与回归

- Model Service 输出无 raw action tensor；
- receipt/reward/accepted/safety 不进入 ObservationSignal；
- Harness 拒绝越界参数和非法 held-input 转移；
- 不完整 checkpoint、flat action array 和旧配置字段 fail closed；
- formal Pretrain、SFT、Online RL 共享同一 Action Head 与概率路径；Online RL 当前使用 Online Recurrent PPO。

## 12. 结论

最终协议把 action 表达成持续的结构化控制流：

```text
H_t + action_local_(t-1)
 -> one Structured ActionFrame
 -> zero or more ordered ControlSignal events
 -> immediate per-unit execution
 -> next physical observation
```

模型像持续操作电脑的人一样，在每个 80 ms unit 观察、行动并从后续物理信号纠错。
统一语义与概率接口并不要求把异构参数语言化；这使短动作保持清晰，也使长 TYPE 不再
付出扁平 token action 序列的结构开销。
