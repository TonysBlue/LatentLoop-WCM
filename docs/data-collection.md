# Harness 驱动的数据采集系统

## 目标与边界

数据采集系统用于生产与最终运行时一致的专家交互轨迹。首版只支持专家扮演助手：用户语音由真实麦克风输入，专家通过采集控制台观察 QEMU、执行电脑操作并说出助手回复。采集器不加载模型，不调用 Model Service，不读取 DOM、可访问性树或隐藏 evaluator 状态。

所有输入都必须经过 Harness 的物理边界。专家动作先编码为当前 `structured-action-v1` 的 `ActionFrame`，再由 Harness 解码为 `ControlSignal`、通过安全门和 QMP 执行。专家语音经 SPICE/虚拟声卡播放，环境声音和播放回流只由真实 mic 采集为唯一 `ObservationSignal.mic`。

## 端到端时序

```text
Task + QEMU snapshot
        |
start_lifetime_session
        |
专家 speech -> SPICE playback -> QEMU audio environment
专家 action -> ActionFrame -> ControlSignal -> Harness/QMP
        |
每 80 ms: capture mixed mic + screen + time
        |
append observation/action/receipt -> hash chain
        |
seal session -> Mimi encode -> current WebDataset episode
```

session 是单一连续 lifetime。第 `t` 个动作的 receipt 与第 `t+1` 个 observation 成对保存；动作、播放和 observation 不允许跨 session 或跳 unit。

## 原始 ledger

`CaptureLedger` 为追加写、崩溃安全的 session 目录：

```text
<session>/
  observation-000000000000.pb
  observation-000000000001.pb
  actuation-000000000000.pb
  receipt-000000000000.pb
  transitions.jsonl
  sealed.json
```

`transitions.jsonl` 保存 unit、payload SHA-256、前序/当前 hash chain、结构化 ActionFrame 和 speech 静音标记。protobuf payload 保存 canonical observation、实际 actuation 和 Harness receipt。`sealed.json` 只有在 unit 连续、TYPE UTF-8 状态闭合且存在至少一个完整 transition 时生成。

## 训练导出

封存后由 `episode_from_capture` 转换成当前 `Episode/StreamUnit`，沿用既有 WebDataset、Mimi 编码、manifest、audit 和 readiness。模型输入仍只有 mixed mic、screen 和 time；专家动作、播放参考、receipt、设备诊断属于监督或审计 metadata。导出样本的 `action_supervision_mask` 为 true，speech mask 按每 unit 的实际播放语音设置。

## 接口与失败策略

- `ExpertCaptureSession.start()` 创建 Harness lifetime 并写入 unit 0 observation。
- `step(ActionFrame, SpeechSignal)` 解码动作、调用 Harness、写入下一 observation 和 receipt。
- `finish(metadata)` 校验 pending UTF-8 已清空并封存 ledger，随后可导出 episode。
- 任意 identity、时序、音频 clock、SafetyGate、SPICE、QMP 或 mic/screen 采集错误都停止 session；未封存目录不得进入 manifest。
- 不提供 fake backend 作为正式回退；fake 只在显式测试中注入。

公共命令读取采集控制台持续写出的 JSONL；`--input -` 表示标准输入。每行必须包含
`action_frame` 和一帧 80 ms 的 `speech_pcm_b64`，没有语音时仍提供全零 PCM 并设置
`speech_silent=true`：

```bash
uv run data capture \
  --config configs/canary.yaml \
  --harness-socket "$HOME/latentloop-data/runtime/canary/harness-control.sock" \
  --snapshot canary-base --session-id expert-001 --lineage-id capture-001 \
  --task-id task-001 --split train \
  --ledger "$HOME/latentloop-data/datasets/canary/capture/expert-001" \
  --output "$HOME/latentloop-data/datasets/canary/shards/staging/sft/sft-%06d.tar" \
  --input - \
  --viewer-socket "$HOME/latentloop-data/runtime/canary/qemu/spice.sock"
```

该命令只生成 `speech_codes_encoded=false` 的 staging shard；进入正式 SFT 前仍必须通过既有
Mimi encode、audit、manifest 和 readiness 路径。

## 运行与验收

专家控制台只负责采集键鼠和麦克风事件，不直接操作 QEMU。QEMU 屏幕通过 WSLg 下的 `virt-viewer` 观察：

```bash
remote-viewer "spice+unix://$HOME/latentloop-data/runtime/canary/qemu/spice.sock"
```

采集验收至少包含：80 ms/24 kHz 对齐、连续 hash chain、动作 receipt identity、动作后屏幕 observation、播放后 mic 回流延迟，以及 QEMU/SPICE/QMP/audio bridge 健康检查。
