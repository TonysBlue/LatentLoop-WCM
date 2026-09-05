# 无模型 Harness 在环数据回放

## 定义

回放器是数据驱动的开环行为回放：不加载 Model Core，不连接 Model Service、Reward Judge 或 Training System。采集 ledger 中的 assistant speech 和 ActionFrame 直接替代模型输出；Harness 仍真实播放语音、执行动作、采集屏幕/mic 并返回 receipt。

```text
CaptureLedger
  +-- speech PCM -> Harness SPICE playback -> QEMU audio -> mic capture
  +-- ActionFrame -> ControlSignal -> Harness/QMP -> screen capture
```

原始 `mic_audio` 不播放。回放首版只实时查看，不覆盖原始数据、不生成新训练样本，也不保存新的 replay episode。

## 时序与控制

每个 80 ms unit 按顺序读取 actuation protobuf，重建 speech 和 ActionFrame，经 Harness `apply` 后等待下一 observation。回放器检查 observation/unit、receipt/unit 连续性；第一个错误 unit 即停止。首版 CLI 支持实时 1x 与 `--non-realtime` 快速回放，不支持暂停、跳转或逐 unit 单步，因为 QEMU 物理时间不会随查看器暂停，伪暂停会破坏回流和屏幕时序语义。

CLI 入口设计为：

```bash
uv run data replay \
  --ledger <capture-session> \
  --harness-socket "$HOME/latentloop-data/runtime/canary/harness-control.sock" \
  --snapshot <snapshot-id> --session-id <replay-id> \
  --viewer-socket "$HOME/latentloop-data/runtime/canary/qemu/spice.sock"
```

也可以直接回放当前 WebDataset episode；这一路要求模型/数据配置仅用于解析样本形状，
不会创建模型：

```bash
uv run data replay \
  --shards "$HOME/latentloop-data/datasets/canary/shards/processed/train/train-*.tar" \
  --config configs/canary.yaml --episode-id <episode-id> \
  --harness-socket "$HOME/latentloop-data/runtime/canary/harness-control.sock" \
  --snapshot <snapshot-id> --session-id <replay-id>
```

CLI 通过 Harness control socket 驱动 session；不接受 model-service socket、checkpoint 或 reward socket 参数。QEMU 屏幕由 WSLg 的 `virt-viewer` 显示，viewer 只观察，不接管输入：

```bash
remote-viewer "spice+unix://$HOME/latentloop-data/runtime/canary/qemu/spice.sock"
```

## 观测与故障

终端每 unit 输出 unit index、receipt accepted、执行延迟、累计时间和下一 observation index。
Safety、QMP、SPICE、音频播放或传感器失败立即终止；不把部分回放结果当作训练数据。
回放生命周期结束后关闭 Harness session 并回收 QEMU overlay/socket。

首版只报告协议、时序和执行状态，不计算“预期画面与实际画面”的语义相似度，也不落盘
实际 observation。这符合实时查看边界；需要可重复的偏差报告时应另行设计审计产物，不能
静默把回放结果写回训练集。

## 验收清单

- 不创建模型实例、不执行模型计算、不连接 Model Service/Reward Judge；读取 shard 时只复用
  当前 `Episode/ActionFrame` 数据类型。
- ActionFrame 经当前 decoder、安全门和 Harness/QMP 执行。
- speech PCM 经真实 playback bridge；原始 mic 不作为播放源。
- 下一时刻 screen/mic 来自 Harness，且 unit 连续。
- 真实模式可在 WSLg/virt-viewer 看到 QEMU；非实时模式有明确标记。
- session/order/UTF-8/设备错误定位到第一个失败 unit。
