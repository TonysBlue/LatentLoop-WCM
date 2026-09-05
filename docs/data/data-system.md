# Data System

Data 是跨 Capture、Replay、Pretrain、SFT、Online Rollout 和 Evaluation 的共享持久化领域。
它管理当前 episode/trajectory/rollout、WebDataset、manifest/hash、Mimi target 编码、
action target、审计和 readiness。数据目录位于仓库外的 `~/latentloop-data/datasets`；大文件、
权重和 socket 不提交 Git。

Canary 是当前唯一正式数据规模，使用统一的 source lock、准备、编码、审计和 readiness 路径：

```text
locked source manifest -> current staging writer -> Mimi worker encode
-> decode/length/hash audit -> atomic processed shards + manifest -> readiness
```

当前输出只写入当前数据目录；不保留历史数据目录或版本化副本。manifest、sample content SHA-256、Mimi
report 和 resolved config 组成 lineage。Canary 缺 source asset 时必须失败，不能生成空或
fixture shard。模型输入只有混合 mic、screen、time 和显式 target mask；decoded action、
receipt、reward、DOM 和 evaluator 私有状态只用于审计或 RL trace。

真实专家轨迹先通过 Harness 采集为追加写 session ledger，再导出当前 staging episode；
具体协议见 [Harness 驱动的数据采集系统](../data-collection.md)。数据检查使用
[无模型 Harness 在环数据回放](../data-replay.md)，其 speech/action 由数据驱动且不调用模型。
