# 直接流式语音（Speech Head）实施说明

> 状态：最终目标语音协议
> 日期：2026-08-20
> 关联顶层架构：[实时流多模态 LatentLoop 完整方案](realtime-multimodal-latent-loop.md)
> 对称动作协议：[统一电脑动作输出协议](unified-action.md)

## 1. 完成边界

Model Service 对外输出 24 kHz mono PCM。Mimi codec token 只存在于 Model Core 的
Speech Head、训练 target 和内部 runtime 解码路径；Harness 只接收 PCM 并负责播放。
SILENCE 由 Model Service 输出全零 80 ms PCM，Harness 不理解 Mimi。

直接语音路径将模型时钟固定为 80 ms。每个 unit 接收一路 24 kHz、1920 样本的混合麦克风输入。完整状态顺序为：

~~~
P_t                 = Perceiver(O_t)
Z_t                 = WorldStateUpdate(Z_(t-1), H_(t-1))
P_hat_(t+1|t)       = Predictor(P_t, Z_t)
F_t                 = PredictionAdapter(stop_grad(P_hat_(t+1|t))) + E_future
H_t, KV_t           = Backbone(P_t, Z_t, F_t, KV_(t-1))
speech_t            = SpeechHead(H_t, speech_local_(t-1))
~~~

Speech Head 每个 unit 预测 SILENCE 或 SPEECH。只有 SPEECH unit 输出一个 Mimi 帧，冻结的因果 decoder 将其转换为 1920 个波形采样。运行路径不经过文本或 TTS；播放回流在下一 unit 作为混合麦克风输入重新进入模型。

## 2. 固定 Codec

| 字段 | 值 |
|---|---|
| Codec | Mimi |
| 采样率 | 24 kHz mono |
| 帧率 | 12.5 Hz |
| 帧长 | 80 ms / 1920 samples |
| Codebook | 8 |
| Vocab | 2048 |
| 上游 revision | `a49141e28b3d9c947cf9aa5314431e1b11cbd2f5` |
| 权重 SHA-256 | `09b782f0629851a271227fb9d36db65c041790365f11bbe5d3d59369cf863f50` |

Mimi decoder 是冻结的声学解码器，不是 TTS。codec identity、revision、权重 hash、采样率、帧率和 codebook 配置必须同时写入数据 manifest、配置和 checkpoint。

codec runtime 通过独立 worker 提供 health、reset、encode_step 和 decode_step 接口。请求必须校验 protocol、session、shape、dtype 和 codec identity；worker 重启时恢复 decoder 的局部连续状态。

## 3. Speech Head

Speech Head 使用一个 learned speech query cross-attend 当前完整 16-slot hidden，再结合上一
时刻 speech local state。它不读取某个固定 token 位置，也不把 Perceiver slot 解释为视觉网格：

~~~
speech_context_t = Attention(speech_query, H_t)
speech_context_t + speech_local_(t-1)
    -> speech mode logits
    -> causal/factorized codec logits
    -> generated Mimi codes
~~~

- SILENCE：只计算 mode CE，codec logits 被 mask，不调用 codec decoder；
- SPEECH：计算 mode CE 和 8-codebook codec CE，并解码为波形 chunk。

Speech Head 的 temporal state 和 previous codes 只负责相邻声学帧连续性，不是认知记忆。codec teacher forcing 只影响帧内预测，不改变跨 unit 的 Z/H/KV 状态转移。推理可以使用 greedy 或配置的采样。

## 4. 数据契约

### 4.1 当前数据契约

每个 episode 主要包含：

~~~
meta.json
mic.flac
target_speech.flac
screen.npz
timeline.npz
speech_codes.npy
turns.json
~~~

每个 unit 的语音 target 为：

~~~
speech_mode          [B]
speech_mode_mask     [B]
speech_codes         [B, 1, 8]
speech_codec_mask    [B, 1]
~~~

`mic.flac` 是模型唯一音频输入。`target_speech.flac` 只用于离线编码、审计和评测；纯静音 unit 的 codec mask 为 false。旧的 `controls.npy`、SpeechControl 状态标签不属于当前协议；历史数据已清理。

### 4.2 编码和校验

数据准备先生成 staging shard，再用锁定 Mimi codec 生成 processed shard。训练前必须验证 schema、codec revision、权重 hash、manifest/content hash 和 session split。目标音频可以由授权 TTS 在数据准备阶段生成，但运行时不出现 TTS 路径。

## 5. 训练目标

语音相关目标只有：

~~~
L_speech = L_speech_mode + L_speech_codec
~~~

`L_speech_mode` 对有效 SILENCE/SPEECH 标签计算 CE；`L_speech_codec` 只对 SPEECH unit 的有效 frame/codebook 计算 CE。没有独立 SpeechControl、prosody、boundary、memory 或 write loss。

当前和未来 speech loss 通过 Speech Head、Backbone、Perceiver、Future Adapter/Gate 以及
TBPTT 内的 WorldStateUpdate 传播。Future 分支在 Predictor 输出处 stop-gradient，因此
speech loss 不训练 Predictor；Predictor 只由 JEPA loss 训练。长期路径为：

~~~
future speech loss
  -> future H
  -> future Z
  -> WorldStateUpdate
~~~

监督长期记忆是否保留有用信息。

## 6. 运行时与验收

运行时必须保证：

- 每个 80 ms unit 最多对应一帧 Mimi codec；
- SILENCE unit 不调用 codec decoder；
- codec token 不直接回灌主干；
- decoder/playback queue 有界；
- codec worker 和 checkpoint 使用同一 identity；
- 连续解码无 NaN、削波、帧漂移和不可接受边界突变；
- Speech mode、每个 codebook accuracy 和静音误触发率可评测；
- Speech Head、Backbone、Perceiver、Future Adapter/Gate 和 WorldStateUpdate 的梯度路径正常；
- speech loss 不越过 predicted slots 的 stop-gradient 进入 Predictor。
