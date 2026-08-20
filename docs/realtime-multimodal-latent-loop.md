# 实时流多模态 LatentLoop 完整方案

> 状态：项目顶层最终架构文档
> 日期：2026-08-20
> 目标：构建持续接收真实混合麦克风和屏幕流、直接生成语音并控制电脑的 always-on 全双工多模态模型。
> 专项协议：[直接流式语音实施说明](direct-speech.md) · [统一电脑动作输出协议](unified-action.md)
> 训练协议：[统一三阶段训练架构](three-stage-training.md) · [Online RL：Online Recurrent PPO 与真实隔离电脑环境](online-recurrent-ppo-training.md)

## 1. 方案概述

### 1.1 系统边界

LatentLoop 由 Model Core、Model Service、Training System、Harness System 和共享
Data 组成。Model Service 的数据平面只接受物理信号并返回物理输出：

```text
Harness sensors -> mic PCM + screen pixels + time
                 -> Model Service
Model Service   -> speech PCM + decoded ControlSignal
                 -> Harness actuators
```

Model Core 内部使用 Speech Head 的 Mimi token 和 Unified Action Head 的结构化
ActionFrame；这些模型输出不作为 Model Service 与 Harness 的直接执行接口。Model Service
将它们解码为 speech PCM 和 ControlSignal 后再发送给 Harness。Harness 不读取或修改
`Z_t/H_t/KV_t`，Training System 不把 reward、receipt 或隐藏环境信息注入模型输入。

共享 Data 负责 capture、replay、监督 episode、online rollout、manifest、审计和
readiness，不隶属于 Training System。

实时流多模态 LatentLoop 是一个运行在真实环境反馈闭环中的递归多模态模型。模型以 80 ms 为一个统一时间单元，持续接收单路混合麦克风音频、屏幕输入和当前观察的时间间隔，通过有界逐层 KV Cache 保存近期精确历史，通过固定容量的抽象 latent workspace `Z_t` 保存长期任务状态，并使用独立 Speech Head 与 Unified Action Head 并行输出。`Z_t` 是模型内部的递归表示，不要求逐一对应真实世界的物理变量，也不要求遵循严格的连续时间动力学。

每个时间单元严格执行：

~~~
P_t                 = Perceiver(O_t)
Z_t                 = WorldStateUpdate(Z_(t-1), H_(t-1))
P_hat_(t+1|t)       = Predictor(P_t, Z_t)
F_t                 = PredictionAdapter(stop_grad(P_hat_(t+1|t))) + E_future
H_t, KV_t           = Backbone(P_t, Z_t, F_t, KV_(t-1))
U_t                  = (
                         SpeechHead(H_t, speech_local_(t-1)),
                         ActionHead(H_t, action_local_(t-1))
                       )
O_(t+1)              = Environment(O_t, U_t)
~~~

H_t 是主干经过 final normalization 后的完整 16-slot hidden 序列，必须暂存到下一单元；
不存在独立 Reasoner。Predictor 的输出不是递归状态，Future slots 也不直接写入 KV。
环境执行结果只通过下一 unit 的真实混合音频和屏幕输入返回模型，不存在显式 ActionEncoder。

## 2. 设计目标

1. 连续接收混合麦克风和屏幕输入，支持长期 always-on 运行。
2. 模型输出语音和电脑 action 时仍持续更新 H、KV、Z 和局部状态。
3. 支持用户插话、补充、纠正和打断，真实回流进入后续 unit。
4. 直接从多模态主干 hidden 生成语音 codec，不经过文本或 TTS。
5. 将所有电脑操控统一到一个结构化 ActionFrame schema 与联合概率接口。
6. 使用固定容量 Z_t 保存长期目标、约束、计划和环境状态。
7. 使用有界 KV 保存近期精确多模态历史，控制显存和延迟。
8. 让未来 Speech/Action loss 通过 Z_t 监督早期 WorldStateUpdate。
9. 训练、验证、checkpoint 恢复和推理使用同一状态转移。
10. 由 Harness 提供动作语法、安全和权限校验。
11. Pretrain、SFT、Online RL（算法为 Online Recurrent PPO）使用同一双头模型和状态转移完整训练全模型；
    RL Value Head 仅用于 actor-critic 估值，不跨物理信号边界。
12. 使用单参数 Perceiver 和 JEPA 单步目标学习可预测的观测表示，并通过门控 Future
    cross-attention 为行为主干提供未来先验。

## 3. 输入与输出

### 3.1 单路混合麦克风输入

模型接收设备实际采集的一路混合音频：

$$
x_t^{mic}=u_t+e_t+o_t+n_t
$$

其中：

- u_t：用户或现场说话人的语音；
- e_t：模型语音经过声卡、扬声器、房间和麦克风后的回流；
- o_t：其他人、电视、音乐、电脑提示音等声音；
- n_t：环境噪声、设备噪声和声学失真。

模型不接收播放参考、来源分离通道或用户/模型语音标识。

### 3.2 屏幕视觉输入

每个 80 ms unit 输入一帧完整的 224x224 RGB 屏幕。静态和动态画面走同一 CNN 与
Backbone 路径；采集缺帧时输入黑帧，不引入有效标志、revision 或变化区域旁路。

### 3.3 直接语音输出

Speech Head 直接预测 Mimi codec token：

~~~
H_t + speech_local_(t-1)
    -> speech mode
    -> Mimi codec frame
    -> frozen causal codec decoder
    -> 1920-sample waveform
    -> playback
~~~

每 80 ms unit 输出 SILENCE 或 SPEECH。SILENCE 不生成 codec token，也不调用 codec decoder。

### 3.4 电脑动作输出

Unified Action Head 每 80 ms 输出一个结构化 frame：

~~~
kind = NO_ACTION | NOOP | POINTER_MOVE | POINTER_BUTTON |
       SCROLL | TYPE | HOTKEY
parameters = kind-conditioned coordinate/button/scroll/text/key fields
~~~

统一的是语义 schema、执行边界和 frame joint probability，不强迫异构参数共享扁平 token
序列。Model Service 把 frame 解码为零个或多个有序 ControlSignal；Harness 校验 schema
和安全策略后在当前 unit 立即执行。

### 3.5 文本旁路边界

文本可以作为离线审计、调试、字幕或数据准备工具，但不是运行时输出头，也不是 Speech Head 的中间目标。生产推理链路只有 Speech Head 和 Unified Action Head。

## 4. 核心状态

设系统按 unit t 运行：

| 符号 | 含义 |
|---|---|
| O_t | 当前混合音频、屏幕和时间观测 |
| P_t | Perceiver(O_t) 的 16 个多模态 slots |
| P_hat_(t+1\|t) | Predictor 对下一时刻 Perceiver slots 的预测 |
| F_t | stop-gradient 后经过适配的独立 Future slots |
| KV_t | 有界逐层 Transformer Key/Value Cache |
| Z_t | 固定容量抽象 latent workspace |
| H_t | 当前 unit 的完整 final-normalized hidden |
| speech_local | 语音 temporal state 和上一帧 codec |
| action_local | previous frame、TYPE decoder、pending UTF-8 和 held-input state |

### 4.1 KV Cache

每层 KV 保存最近进入主干的 16 个 Perceiver slots。缓存按完整 unit 追加和淘汰，
不能在 unit 中间截断。生产上限为 750 个 80 ms unit，即每层 12,000 tokens、60 秒。
World slots 和 Future slots 不作为额外 token 直接写入 KV。

### 4.2 Latent memory

$$
Z_t\in\mathbb R^{B\times M\times d_z}
$$

Z_t 由固定数量的 slots 构成，用于保存：

- 用户目标和长期约束；
- 当前任务阶段和未完成子目标；
- 应用、窗口和重要 UI 状态；
- 动作计划、失败恢复和安全状态；
- 已经离开 KV 窗口但仍影响未来输出的信息。

Z_t 的容量与运行时长无关，不承诺逐 token 复制历史。

### 4.3 完整 H_t

$$
H_t\in\mathbb R^{B\times 16\times d_{model}}
$$

H_t 是当前 unit 的完整主干输出，而不是单个 query 或 pooled summary。它必须保存在 RecurrentState.hidden，并作为下一时刻 WorldStateUpdate 的唯一 hidden 输入。

### 4.4 局部状态

speech_local 只维护相邻 codec 帧的声学连续性；action_local 只维护未结束 action event 的 decoder 连续性。二者不是长期认知记忆，不能替代 Z_t。

### 4.5 状态初始化和边界

episode/session 开始时：

- Z_0、H_0、audio cache、speech_local、action_local 清零；
- KV_0 为空；
- unit_index 从零开始。

正常 unit 边界不重置状态。TBPTT 边界只 detach 计算图，不清空状态数值。

## 5. 多模态时间单元

### 5.1 Unit 格式

每个 unit 的输入协议为：

~~~
timestamp_ms       [B] int64
delta_ms           [B] int64
mic_audio          [B, 1920] float32
screen             [B, 3, 224, 224] float32
~~~

delta_ms 必须为正，时间戳严格递增。

### 5.2 Perceiver 输入与输出

音频、视觉和时间 stem 先组织为带模态与位置编码的输入序列：

~~~
<TIME>
<AUDIO_0> ... <AUDIO_N>
<VISION_0> ... <VISION_15>
~~~

16 个 learned Perceiver queries 通过两层 `slot-to-input cross-attention -> slot
self-attention -> FFN` 将该序列压缩为 `[B,16,model_dim]` 的 P_t。视觉 stem 仍输出
4x4 空间特征，但 Perceiver slots 是混合模态高层表示，不再与视觉位置一一对应。所有
观测必须经过 Perceiver，原始 stem token 不得绕过它进入 Backbone。静态和动态屏幕每个
unit 都走同一条编码路径。

### 5.3 Codec 时间对齐

~~~
sample_rate          24000 Hz
unit_ms              80
audio_samples/unit   1920
codec_frame_rate     12.5 Hz
codec_frames/unit    1
codebooks            8
vocabulary           2048
~~~

音频和 codec 时间轴必须严格对应，不能累计四舍五入。

### 5.4 多频率调度

采集器可以高频采集音频和屏幕，编码器可以使用内部帧率，但主干状态转移和输出协议统一在 80 ms unit 上。任何降频、合并或背压都必须显式记录 delta_ms，并保持状态顺序。

### 5.5 时间间隔编码

模型不直接接收绝对时间。`timestamp_ms` 只用于 unit 排序、计算 `delta_ms`、延迟统计和
轨迹记录。当前观察间隔通过 DeltaTimeEncoder 进入 Backbone：

$$
T_t^{\Delta} = \mathrm{DeltaTimeEncoder}(\Delta t_t)
$$

DeltaTimeEncoder 使用 log-scaled interval 和多尺度 Fourier 特征：

$$
\phi_k(\Delta t) =
\left[
\sin\left(2\pi\frac{\Delta t}{p_k}\right),
\cos\left(2\pi\frac{\Delta t}{p_k}\right)
\right]
$$

其中周期集合默认为 `80, 160, 320, 640, 1280, 2560, 5120, 10240 ms`。时间间隔不进入
WorldStateUpdate，也不要求 `Z` 遵循物理时间动力学。

## 6. 模型架构

~~~
MIC_MIXED --> Streaming Audio Encoder --┐
SCREEN   --> Vision Encoder -----------+--> Perceiver --> P_t
DELTA_T  --> DeltaTimeEncoder ---------┘                 |
                                                         +--> Predictor --> P_hat_(t+1|t)
Z_(t-1), H_(t-1) --> WorldStateUpdate --> Z_t -----------+         |
                                                                  stop_grad
                                                                      |
                                                        PredictionAdapter --> F_t
                                                                      |
KV_(t-1), P_t, Z_t, F_t --> Backbone --> H_t, KV_t
                                      |             |
                               Speech Head       Action Head
                                      |             |
                             Mimi waveform     ActionFrame
~~~

### 6.1 流式音频编码器

音频编码器增量处理新增采样并保留 audio_cache。它只接收混合麦克风，不接收播放参考或来源标签。

### 6.2 视觉编码器

视觉编码器是纯 PyTorch 轻量残差 CNN，从 `224x224` RGB 屏幕帧输出 `[B,16,model_dim]`。
16 个 token 对应 4x4 空间网格并带可学习二维位置编码；不使用全局池化为单 token，也不
区分静态和动态画面。缺失屏幕帧在输入适配器中变为全黑帧，仍按正常 unit 推进时间和状态。

视觉 token 只作为 Perceiver 输入。Action Head 使用一个 learned state query 和 4x4 learned
spatial queries cross-attend 完整 H_t，再预测 ActionFrame；它不能把 16 个混合模态
Perceiver slots 当作屏幕网格，也不存在 VisionEncoder 到 ActionHead 的旁路。

KV 不再区分视觉和非视觉类别。每层按 unit 顺序追加 16 个当前 slots，统一保留最近
`kv_units` 个 unit，并由 checkpoint 和 TBPTT detach 一起维护。

### 6.3 多模态主干

主干执行：

$$
H_t,KV_t=F_\theta(P_t,Z_t,F_t,KV_{t-1})
$$

同一 unit 的 16 个 slots 双向互见，只能读取历史 unit 的 KV。每层依次执行 cached
self-attention、可选 World cross-attention、可选 gated Future cross-attention、feed-forward
和 normalization。Z_t 通过独立投影/cross-attention 进入主干，不拼接为普通 token KV。

Future 分支定义为：

$$
Q_t^{(l)}=\mathrm{LayerNorm}(H_t^{(l)}),\qquad
C_t^{(l)}=\mathrm{FutureCrossAttention}(Q_t^{(l)},F_t,F_t)
$$

$$
G_t^{(l)}=\sigma\left(W_g^{(l)}[Q_t^{(l)},C_t^{(l)}]+b_g^{(l)}\right)
$$

$$
H_t^{(l)}\leftarrow H_t^{(l)}+G_t^{(l)}\odot C_t^{(l)}
$$

G 的形状为 `[B,16,1]`。W_g 零初始化、b_g 初始为 -4，使训练初期预测先验只以小残差
进入主干。Gate 不重复直接读取 P_t 或 Z_t；当前 hidden 已经包含当前与 World 上下文。

### 6.4 Predictor 与单参数 JEPA

Predictor 读取 P_t 和 Z_t，以固定 slot 顺序预测一个 80 ms 后的 Perceiver 表示：

$$
\widehat P_{t+1|t}=\mathrm{Predictor}(P_t,Z_t)
$$

Perceiver 只有一份参数，不维护 EMA teacher。预测输出一方面以未截断形式参与 JEPA loss，
另一方面在 stop-gradient 后经过 PredictionAdapter 和 learned future embedding 形成 F_t。
行为 loss 因此训练 Adapter 和 Future Gate，但不会沿该分支进入 Predictor。

## 7. LatentLoop 状态更新

### 7.1 严格更新顺序

WorldStateUpdate 先于当前 unit Backbone：

$$
Z_t = U_\theta\left(Z_{t-1}, H_{t-1}\right)
$$

`Z_t` 是唯一实际递归的 latent 状态。文档中可以用“预测”和“吸收观察”解释这个变化过程，
但实现不拆分 `Z_t` 为预测状态和校正状态，也不维护两套 latent。WorldStateUpdate 不接收
`delta_t`、当前音频、当前视觉、Action 或 Speech 输出；当前 unit 的观察和时间间隔先进入
Backbone，形成的 `H_t` 在下一轮影响 `Z_(t+1)`。

### 7.2 候选与门控

一种等价内部参数化为：

$$
\begin{aligned}
Q_{t-1} &= W_q(Z_{t-1}) + I_{slot},\\
C_{t-1} &= \mathrm{Attention}(Q_{t-1}, H_{t-1}, H_{t-1}),\\
\Delta Z_{t-1} &= \mathrm{Candidate}(Z_{t-1}, C_{t-1}),\\
G_{t-1} &= \sigma\left(\mathrm{Gate}(Z_{t-1}, C_{t-1}) - 2\right),\\
Z_t &= \mathrm{LayerNorm}\left(Z_{t-1} + 0.1 G_{t-1}\odot\Delta Z_{t-1}\right).
\end{aligned}
$$

`G_(t-1)` 按 slot 和 latent dimension 生成，而不是单个全局标量。门控残差只是状态更新的
稳定参数化，不表示 `Z` 遵循一个物理微分方程。

learned slot identity 只打破零初始化 slots 的对称性，不规定 slot 语义。

### 7.3 写入和遗忘

门控残差允许模型在信息不重要时保持旧状态，在目标变化或新事实出现时写入候选。重要信息的相对影响会在后续任务中被增强，不重要信息会因后续更新和有限容量而逐渐相对减弱。没有人工 write-budget 或 diversity loss。

### 7.4 长时监督

未来输出 loss 沿以下路径反向传播：

~~~
future Speech/Action loss
  -> future H
  -> future Z
  -> earlier WorldStateUpdate
  -> earlier H and Z
~~~

长期记忆是否有效，以跨窗口 Speech/Action 行为评测和 latent on/off 消融为准。

## 8. 直接语音生成

### 8.1 Speech Head

Speech Head 使用 H_t 的当前输出位置和 speech_local：

~~~
H_t + speech_local_(t-1)
    -> speech mode logits
    -> causal/factorized codec logits
    -> generated codes
~~~

Speech mode 为 SILENCE 或 SPEECH。SPEECH unit 预测一帧 8-codebook Mimi token；SILENCE unit 的 codec mask 为 false。

### 8.2 Codec 契约

| 字段 | 值 |
|---|---|
| Codec | Mimi |
| 采样率 | 24 kHz mono |
| 帧率 | 12.5 Hz |
| 帧长 | 80 ms / 1920 samples |
| Codebook | 8 |
| Vocab | 2048 |
| Revision | 配置锁定 |
| Weight SHA-256 | 配置锁定 |

Mimi decoder 是冻结的声学解码器，不是 TTS。

### 8.3 环境自听

codec 解码和播放后的声音经过真实扬声器、房间和麦克风回流：

$$
Y_t\rightarrow Playback\rightarrow Environment\rightarrow x_{t+\delta}^{mic}
$$

回流作为下一 unit 的混合音频重新进入 Perceiver，不直接把 codec token 回灌主干。

## 9. Unified Action Head 与 Harness

### 9.1 Structured ActionFrame

kind 固定为 `NO_ACTION/NOOP/POINTER_MOVE/POINTER_BUTTON/SCROLL/TYPE/HOTKEY`。
POINTER_MOVE 使用 32x32 joint coarse cell categorical 与 cell 内 bounded residual；
POINTER_BUTTON 使用 button 与 `CLICK/DOWN/UP` phase 且作用于当前指针；SCROLL 使用二维
bounded delta；TYPE 使用每 unit 至多 16 bytes 的 UTF-8 decoder；HOTKEY 使用版本化 key
decoder。只有当前 kind 对应的参数参与 loss 与 frame joint log-prob。

### 9.2 组合动作

拖拽由 `button DOWN -> move* -> button UP` 组成，双击由两个 CLICK frame 组成，等待由
NO_ACTION 表达。宏动作、duration、CANCEL、END_ACTION 和 PAD 均不属于协议。

### 9.3 跨 unit TYPE

连续 TYPE frame 隐式续接。action_local 最多保存 3 个 pending UTF-8 bytes，并在每个 unit
把已经完成的合法文本前缀立即解码、校验和执行；切到其他 kind 时结束 TYPE 且 pending
必须为空。已经执行的 action 不回滚，模型通过后续真实屏幕/声音继续纠正。

### 9.4 Harness 安全边界

Harness 在执行前校验：

- schema 和 kind-conditioned 参数完整性；
- 坐标、滚动、文本、key 和 held-input 状态范围；
- 应用和区域白名单；
- 删除、支付、发送、安装和权限修改审批；
- 速率限制、session reset 和全局紧急停止。

执行结果只通过下一 unit 的屏幕和声音反馈进入模型。

## 10. 并发输出

语音和 action 可以在同一 unit 并行产生，但两者保持独立输出空间：

~~~
H_t -> Speech Head -> SILENCE/SPEECH + codec
H_t -> Action Head -> one Structured ActionFrame
~~~

不存在独立 Speech Control、Action Control 或 Cognitive Control head。静音由 Speech Head
的 SILENCE mode 表达，等待由 Action Head 的 NO_ACTION 表达；session reset 和紧急停止是
Harness control-plane 操作，不是模型 action kind。

## 11. 完整状态转移

### 11.1 感知与预测

$$
P_t=Perceiver(O_t),\qquad
\widehat P_{t+1|t}=Predictor(P_t,Z_t)
$$

### 11.2 记忆

$$
Z_t=WorldStateUpdate(Z_{t-1},H_{t-1})
$$

### 11.3 主干

$$
F_t=PredictionAdapter(stop\_grad(\widehat P_{t+1|t}))+E_{future}
$$

$$
H_t,KV_t=Backbone(P_t,Z_t,F_t,KV_{t-1})
$$

### 11.4 输出

$$
U_t=\left(
SpeechHead(H_t,speech\_local_{t-1}),
ActionHead(H_t,action\_local_{t-1})
\right)
$$

### 11.5 状态保存

$$
state_{t+1}=(Z_t,H_t,KV_t,audio\_cache_t,speech\_local_t,action\_local_t)
$$

### 11.6 环境演化

语音播放和 action 执行改变真实环境；其后续麦克风、屏幕和时间输入构成 O_(t+1)。
模型不读取隐藏的执行成功标签，也不把 U_t 作为显式 Predictor 条件。

## 12. 上下文管理

### 12.1 有界 KV

KV 统一保留最近配置窗口：

~~~
KV_t = CURRENT_SLOTS[t-kv_units+1:t]
~~~

生产上下文为 750 units（60 秒），每个 unit 固定 16 个 slots。KV 按完整 unit 淘汰，
保留后的 token 仍按原始时间顺序参与 causal attention。

### 12.2 Latent memory 读取

Z_t 在主干指定层通过 latent cross-attention 读取。Z_t 不并入普通 KV，不随 KV 淘汰被删除；只有 episode/session reset 才清零。

### 12.3 持久化

checkpoint 保存 Z、H、KV、audio cache、speech local、action local 和 unit cursor。会话持久化必须记录当前 model、codec 和 action identity；不完整身份拒绝恢复。

## 13. 实时运行时

### 13.1 运行线程

~~~
Audio Capture       音频环形缓冲
Screen Capture      每 unit 完整屏幕帧
Perceiver           音频/视觉/时间 stem 与 16-slot 感知编码
Backbone Worker     WorldStateUpdate、Backbone、KV/Z 状态
Speech Worker       codec frame 和播放块
Action Worker       Harness grammar/safety/execution
Telemetry           延迟、队列、状态和轨迹
~~~

### 13.2 单 GPU 调度

递归模型、KV、Z、audio cache 和两个 local state 留在同一个 GPU 进程。Ray 只负责外围 CPU 任务，不逐 unit 搬运状态。

### 13.3 背压

当计算延迟超过实时 tick：

- 音频块可以合并，但必须更新 delta_ms；
- 视觉按 unit 输入当前完整帧；采集缺帧填充黑帧并记录 telemetry；
- 播放队列和 action 队列保持有界；
- action 队列保持有界并按 unit 顺序执行；
- 超时、取消和紧急停止优先级最高；
- 任何丢帧、时间跳跃或 worker 错误都写入 telemetry。

## 14. 训练数据

### 14.1 轨迹字段

训练轨迹由以下数据构成：

~~~
mic_audio          单路混合麦克风
screen_frames      每 unit 一帧完整屏幕
target_speech      仅用于离线 codec 编码的目标音频
target_actions     每 unit 一个结构化 ActionFrame
timestamps         unit 时间
runtime_events     播放、执行、丢帧和延迟审计
~~~

构造过程中的来源分离、TTS 中间产物、环境参数和任务标签不能作为模型输入。

当前 WebDataset episode 由 `meta.json`、`mic.flac`、
`target_speech.flac`、`screen.npz`、`timeline.npz`、`speech_codes.npy` 和
`turns.json` 组成；timeline 保存 speech mode/mask、codec mask、结构化 action frame、
action supervision mask 和时间戳。`controls.json` 和 `receipts.json` 只保存
control-plane 审计，不进入模型输入。旧 flat `action_tokens/action_token_mask`、
`controls.npy` 和 memory target 不属于当前训练协议；历史资产已清理，后续只从源轨迹
构建当前数据。完整字段见
`docs/data/trajectory-schema.md`。

### 14.2 场景覆盖

最终数据集应覆盖：

- 安静单人语音；
- 用户打断、重叠语音和多人环境；
- 键盘、风扇、音乐、电视和系统声音；
- 播放延迟、混响、设备频响、回流衰减和回流缺失；
- 静态屏幕、窗口切换、动态 UI 和连续运动目标；
- 点击、拖拽、滚动、输入、快捷键、等待、取消；
- 动作失败、用户纠正、长任务和跨应用任务；
- 长时间无语音、纯屏幕观察和模型静音。

### 14.3 声学域与视觉域随机化

训练可以随机化房间 impulse response、设备频响、播放延迟、时钟漂移、非线性失真、自动增益、背景噪声、回流截断和视觉变化速度，但最终输入仍必须是一路混合麦克风和屏幕 unit。

### 14.4 数据隔离

按 device/session 分组切分 train/validation/test。同一 session 不得跨 split。每个 manifest 锁定 source、license、content hash、codec identity 和 session hash。

## 15. 训练目标

### 15.1 Speech loss

$$
L_{speech}=L_{speech\_mode}+L_{speech\_codec}
$$

mode loss 对有效 SILENCE/SPEECH 标签计算 CE；codec loss 只对 SPEECH unit 的有效 Mimi frame/codebook 计算 CE。

### 15.2 Action loss

$$
L_{action}=-E[\log p(ActionFrame_t\mid H_t,action\_local_{t-1})]
$$

frame joint log-prob 由 kind categorical 与对应的 coordinate/button/scroll/text/key 参数项
组成。各分支只在有效 kind 和监督 mask 上归一化；连续参数的训练 NLL 与 sampling
log-prob 使用同一 bounded 参数化。

### 15.3 并发输出

Speech 和 Action 共享 Backbone 梯度，但使用独立 loss 和独立输出 vocabulary。一个 unit 可以同时具有有效 Speech 和 Action target。

### 15.4 长期记忆监督

Z 没有独立 memory loss、probe loss、write-budget 或 diversity loss。Predictor 使用独立
JEPA 表示目标，但它不是 Z 的 memory target。未来 Speech/Action loss 仍通过：

~~~
future loss -> future H -> future Z -> earlier WorldStateUpdate
~~~

监督 Z_t 的长期信息选择。

### 15.5 JEPA 表示预测损失

同一个参数版本同时计算 source 和 target，target 侧停止梯度：

$$
L_{pred}=E_{t,s}\left[
\left\|N(\widehat P_{t+1|t,s})-stop\_grad(N(P_{t+1,s}))\right\|_2^2
\right]
$$

其中 N(X)=X/max(||X||_2,10^-4)。为防止单参数 Perceiver 表示坍塌，在有效 source
表示上按 slot、channel 跨 batch-time 计算：

$$
\sigma_{s,d}=\sqrt{Var_{b,t}(P_{t,s,d})+10^{-4}},\qquad
L_{var}=E_{s,d}[max(0,1-\sigma_{s,d})]
$$

$$
L_{JEPA}=L_{pred}+L_{var}
$$

有效 source 少于两个时跳过 L_var；不得跨 episode、session 或不连续 observation 配对。

### 15.6 总损失

Pretrain、SFT 和 Online RL 分别使用 1.0、0.5 和两路 0.1 的 JEPA 系数，完整公式由
[统一三阶段训练架构](three-stage-training.md) 约束。各模块影响关系为：

| 模块 | 直接梯度 | 主要行为影响 |
|---|---|---|
| Speech Head | Speech mode/codec loss | 语音 mode、codec 准确率和局部连续性 |
| Action Head | Action token loss | grammar、参数 token 和跨 unit continuation |
| Perceiver | Speech、Action/RL、JEPA source loss | 当前多模态观察表示 |
| Predictor | JEPA loss | 单步未来表示预测 |
| PredictionAdapter/Future Gate | Speech、Action/RL loss | 受控注入未来先验 |
| Backbone | Speech、Action/RL loss | 共享多模态理解和输出条件表示 |
| WorldStateUpdate/Z | 未来输出 loss 与 JEPA source loss | 长期目标、约束、计划和抽象任务状态保持 |
| KV state | 无参数 loss | 近期精确上下文 |
| local states | 对应 head loss | 语音跨帧和 action 跨 unit 连续性 |

## 16. 闭环训练

### 16.1 训练分布

训练输入中的未来麦克风可以由真实录音、目标语音经过声学环境后的回流或模型实际生成语音经过环境模拟后的回流构成；无论来源如何，模型看到的都必须是同一路混合麦克风。

### 16.2 环境反馈建模

声学环境可包含播放延迟、房间响应、设备频响、噪声、重叠、截断和回流缺失。视觉环境可包含窗口变化、连续运动、动作延迟和动作失败。

### 16.3 梯度路径

单个 unit 的输出 loss 通过当前 Backbone 和 Perceiver 反向传播；跨 unit 的未来行为 loss
通过 Z 和 H 反向传播。JEPA target 侧 detach，source 侧可通过 Predictor、Z 和 TBPTT 内
历史状态传播。TBPTT 只在配置边界 detach，不能在每个 unit 重置状态。

### 16.4 外部执行边界

物理播放、操作系统和 UI-TARS 执行本身不是可微模块。它们提供后续观察和行为 reward；训练接口只接收合法的 input/target/observation，不把隐藏执行结果作为额外模型输入。

## 17. 训练约束

1. Canary、Pilot、Production 使用同一个训练入口和状态协议。
2. 所有训练 episode 按时间顺序处理，不能每个窗口重置状态。
3. 生产 memory horizon 和 TBPTT 为 750 units。
4. 所有 targets 配套 mask；缺失标签屏蔽对应 loss，不伪造 NOOP 或 SILENCE。
5. 未来输出 loss 必须能够在 TBPTT 范围内回传到早期 WorldStateUpdate。
6. 训练、验证、推理和恢复共享同一 forward_step 语义。
7. codec、action schema、trajectory schema、unit 时钟和 checkpoint identity 必须一致。

## 18. 推理算法

~~~
state = initial_state()

for each 80 ms observation O_t:
    P_t = Perceiver(O_t)
    Z_t = WorldStateUpdate(state.Z, state.H)
    P_hat = Predictor(P_t, Z_t)
    F_t = PredictionAdapter(stop_grad(P_hat)) + E_future
    H_t, KV_t = Backbone(P_t, Z_t, F_t, state.KV)
    speech_t = SpeechHead(H_t, state.speech_local)
    action_t = ActionHead(H_t, state.action_local)

    submit speech_t to codec/playback if mode=SPEECH
    submit action_t to Harness if grammar/safety checks pass

    state = {
        Z: Z_t,
        H: H_t,
        KV: KV_t,
        speech_local: speech_t.local,
        action_local: action_t.local,
    }
~~~

输出产生的真实音频和屏幕变化在后续 unit 重新进入输入。不存在 control-head 决定是否运行这两个 head 的额外状态机。

## 19. MiniCPM 系基础实现

MiniCPM 或同类多模态主干可以提供视觉编码、音频编码、Perceiver stems、因果 Backbone
和增量 KV。项目必须在此基础上保持：

1. 固定 80 ms unit；
2. 完整 H_t 暂存；
3. Z_t = WorldStateUpdate(Z_(t-1), H_(t-1))；
4. 独立 Speech Head；
5. Unified Action Head；
6. 单路混合麦克风输入；
7. 有界 KV 和固定 latent slots；
8. 同一 checkpoint/data/trajectory identity。

不得引入与上述状态协议不一致的 control、memory 或 action 过渡接口。

## 20. 配置约束

目标配置必须明确：

- model_dim、latent_dim、层数、heads 和 FFN；
- latent_slots、kv_units、kv_window_ms；
- audio sample rate、unit_ms、screen shape；
- Mimi codec identity；
- action_schema_id、coordinate_grid_size、type_bytes_per_unit、hotkey_keys_per_unit；
- tbptt_units、memory_horizon_units、mixed precision；
- loss weights、checkpoint cadence、manifest 和 run identity。

生产、Canary、Pilot 的正式 horizon 为 750 units；Smoke 只缩小数值，不改变协议。

## 21. 评测与消融

### 21.1 基线与消融

| 方案 | 用途 |
|---|---|
| 有界 KV，无 Z | 近期上下文基线 |
| 有界 KV + Z | 核心长期记忆方案 |
| 长 KV 对照 | 精确历史上界 |
| 核心方案去掉完整 H 暂存 | 验证完整 hidden 的必要性 |
| 核心方案去掉 Action continuation | 验证跨 unit action state |
| 核心方案去掉 Speech codec mask | 验证 SILENCE 语义 |

### 21.2 语音指标

- mode accuracy；
- speech-active fraction；
- 每个 codebook accuracy；
- SILENCE 误触发率；
- codec RTF、首块延迟和长语音连续性；
- 帧漂移、NaN、削波和播放队列积压；
- 自声回流、噪声和重叠条件下的稳定性。

### 21.3 视觉与动作指标

- action kind accuracy 与 frame joint NLL；
- schema validity；
- 坐标 cell 命中/残差误差、button/phase、scroll 和 text/key accuracy；
- TYPE UTF-8 跨 unit continuation；
- 动态目标动作时延与命中率；
- 动作成功率、失败恢复和长任务完成率；
- 危险动作误执行率。

### 21.4 Long-term state 指标

- Z on/off 跨窗口任务差异；
- KV 窗口缩短时的目标保持曲线；
- 用户约束更新和冲突修正；
- 早期 action 结果在后续的正确利用；
- H_t 完整暂存与摘要替代的对照；
- Z、KV、H 和 local state 的容量上界。

### 21.5 实时系统指标

- unit latency、p95/p99；
- 音频、视觉、codec、action 和 playback 队列长度；
- GPU 显存、CPU 内存和 socket 健康；
- 长时间运行中的 NaN、丢帧、underrun 和队列增长；
- checkpoint 恢复后的下一 unit 输出与 loss 一致性。

## 22. 风险与缓解

| 风险 | 缓解策略 |
|---|---|
| Z 被近期 KV 忽略 | 窗口外任务、Z on/off 消融和长期行为评测 |
| Z 过度写入或过度保持 | 依靠未来输出任务、门值监控和状态范数监控 |
| slots 同质化 | learned slot identity、行为消融和容量监控 |
| 声学回流导致重复响应 | 回流延迟/音量/缺失随机化、重复行为评测 |
| 用户插话导致状态错乱 | 真实混合音频、严格 unit 顺序和完整 H 暂存 |
| 动态画面导致动作滞后 | 连续视觉流训练、80 ms unit 时序和运动任务评测 |
| 单卡吞吐跟不上 | 音频合并、视觉降频、activation checkpoint、显式背压 |
| 长时间队列增长 | 有界队列、超时、取消和实时 telemetry |
| 隐状态难审计 | action approval、完整轨迹、checkpoint lineage 和紧急停止 |

## 23. 安全边界

UI-TARS/Harness 必须提供：

- 删除、支付、发送、安装和权限修改审批；
- 应用和区域白名单；
- 坐标、时长和速率校验；
- 全局停止快捷键；
- 操作轨迹记录和回放；
- 输入文本和屏幕媒体脱敏；
- 无人值守策略和失败时的安全默认值。

语音运行时必须提供最大连续输出时长、音量限制、异常重复检测和播放队列上限。模型不能绕过执行层直接调用高风险操作。

## 24. 部署与运行时契约

- 模型、KV、Z、H、codec local state 在同一推理进程内保持；
- W&B 只负责指标、配置和谱系，不进入模型 forward；
- Ray 只负责 CPU 数据、环境和评测，不维护 GPU recurrent state；
- codec worker 通过带身份校验的本地接口提供 encode/decode；
- runtime、checkpoint、manifest、codec 和 action schema identity 必须一致；
- 所有异常通过显式失败、隔离、恢复或安全拒绝处理。

## 25. 成功标准

1. 模型可以持续处理一路混合麦克风和屏幕流。
2. Speech Head 以 80 ms 对齐生成 SILENCE/SPEECH 和 Mimi codec。
3. 自声回流、用户插话、噪声和屏幕变化不会破坏状态顺序。
4. 有界 KV 和固定 Z 的容量不随运行时长无限增长。
5. Z on/off 对跨窗口目标、约束和错误恢复产生可测差异。
6. Unified Action Head 能表达 grammar 合法的电脑操作并跨 unit continuation。
7. Harness 能拒绝过期、越权和危险 action。
8. checkpoint 恢复后的下一 unit 输出和 loss 与连续运行一致。
9. 长时间运行中延迟、队列、显存和 socket 保持有界。
10. 训练、验证、推理和恢复均符合本顶层状态协议。

## 26. 最终架构定义

实时流多模态 LatentLoop 的最终形态为：

~~~
mixed microphone + screen + time
    -> Perceiver(O_t) = P_t
    -> Z_t = WorldStateUpdate(Z_(t-1), H_(t-1))
    -> Predictor(P_t, Z_t) = P_hat_(t+1|t)
    -> PredictionAdapter(stop_grad(P_hat)) = F_t
    -> H_t, KV_t = Backbone(P_t, Z_t, F_t, KV_(t-1))
    -> SpeechHead(H_t) + UnifiedActionHead(H_t)
    -> frozen Mimi decode / Harness execution
    -> real acoustic and visual feedback
    -> next 80 ms unit
~~~

KV 负责近期精确历史，Z_t 负责固定容量长期状态，H_t 负责把当前完整主干结果连接到下一次 memory update。语音和电脑操控共享主干但保持独立输出空间；这就是项目的顶层最终架构。
