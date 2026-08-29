# LatentLoop Agent Instructions

This file is the repository-level operating contract for coding agents and
developers. Read it before changing code, configuration, data tooling, or
training scripts. More specific `AGENTS.md` files, if added later, override
this file only for their directory. System, developer, and user instructions
always take precedence over this document.

## Project Mission

LatentLoop is a research and training harness for an always-on, real-time,
full-duplex multimodal agent. The model consumes one mixed microphone stream,
screen frames, and recurrent state; it directly predicts speech codec tokens
and has an independent action head. The surrounding harness is responsible
for workers, playback, computer control, and deployment integration.

The repository is a research implementation, not a product UI. Prefer
reproducibility, explicit contracts, measurable gates, and small reversible
changes over convenience abstractions.

## Design Target Principle

方案设计、接口和训练目标应直接面向最终优雅、统一且自洽的目标架构，
不要为了短期可运行而引入与目标架构语义不同的中间妥协方案。讨论或实现
新方案时，先明确最终状态、数据流、输出空间和监督路径；除非用户明确要求
过渡实现，否则不添加临时分支、占位 head 或仅服务于中间阶段的设计。

涉及架构、状态、数据协议、训练目标或 checkpoint 的变更，必须先刷新
设计文档并完成测试设计，再开始生产代码实现。除非用户明确要求过渡
实现，不得先写中间版本再回补最终方案。

## Core Principle: Same Code Path

Canary is the only formal training scale while local validation is in progress.
Future scale design must follow from Canary evidence and update the architecture,
test contract, configuration, and asset plan before adding another formal profile.

- All optimizer training goes through the Training System's shared `train` dispatcher.
- Multi-stage execution goes through the Training System's shared recipe runner.
- The public orchestration wrapper is `scripts/run-training.sh`.
- Evaluation is owned by the Training System and uses the shared
  `evaluate_checkpoint` implementation.
- Data preparation goes through `data`/`scripts/data/prepare.sh` and the shared
  curation modules.
- Do not add a stage- or scale-specific training loop.
- E1/E2 and similar milestone labels are process terminology only. Do not put
  them in source filenames, package names, configuration keys, artifact
  directories, or CLI commands. They may appear in a design document's plan.

When a stage needs new behavior, first add a configuration field or a shared
capability with tests. Only add stage-specific branching when the data/model
contract genuinely differs, and document why in the design documentation.

正式训练 recipe 的阶段语义固定为 `Pretrain -> SFT -> Online RL`。当前只保留
Canary 正式规模，并完整执行这三个阶段，共享模型、训练循环、真实隔离电脑环境协议、
reward 定义和 checkpoint 谱系；Online RL 的当前正式算法固定为 Online Recurrent PPO。
后续规模必须等本机 Canary 验证形成数据、显存、吞吐和质量证据后重新设计，不预留未验证
的规模配置或资产入口。正式单智能体 Online RL 始终只有一个 active
lifetime session，不以环境并发分叉时间线。不得用 fixture、
离线 rollout、critic/value head 或只训练输出 head 来替代任何正式阶段。测试可以
使用显式标记的进程内环境实现协议契约，但该实现不得成为正式配置的回退路径。

Pretrain 和 SFT 都训练 InputEncoder、Backbone、MemoryUpdater、Speech Head 与
Unified Action Head。Online RL 使用 Online Recurrent PPO：冻结的 SFT reference policy、单一生命期
ObservationSignal 时间线、感知 Reward Event、time-discount GAE 和 clipped policy ratio，
全模型接受策略梯度；训练专用 Value Head 不跨 Model Service 物理边界，也不新增独立
memory loss。所有优化仍由
`training.py::train` 分派，所有阶段仍由 `recipe.py::run_recipe` 编排。

## Repository Map

### System boundaries

The final architecture has three runtime systems and shared packages:

- `Model Core` is pure tensor/model code. It owns the `Z/H/KV` recurrent state,
  Backbone, Speech Head, Unified Action Head, and token sampling.
- `Model Service` is the physical-signal inference boundary. It accepts mixed
  microphone PCM, screen pixels/revision, and time; it returns speech PCM and
  decoded `ControlSignal` events. It never captures devices or executes OS
  input.
- `Training System` owns Pretrain, SFT, Online RL, replay, optimizer,
  checkpoint lineage, and evaluation. It uses the same Model Core as serving.
- `Harness System` owns microphone/screen capture, playback, isolated computer
  lifecycle, ControlSignal validation/safety, execution, receipts, and reward.
- `Data` is a shared persistence domain for capture, replay, supervised
  episodes, rollout traces, manifests, and readiness. It is not private to the
  training package.

The Model Service data plane is strictly physical signals:

```text
ObservationSignal: mic PCM + screen pixels/revision + time
ActuationSignal:   speech PCM + decoded ControlSignal events
```

Action tokens remain an internal Model Core/training/data representation and
are decoded before crossing the Model Service-to-Harness execution boundary.
Rewards, receipts, task success, DOM/accessibility data, and other hidden
environment fields are control-plane or training metadata only; they never
enter `ObservationSignal`.

The target source layout is a uv workspace with `packages/contracts`,
`packages/model`, `packages/media`, `packages/data`, `packages/runtime`, and
`systems/model-service`, `systems/training`, `systems/harness`. The legacy
single package is migrated into these domains in one coherent change; no
stage-specific implementation or fixture fallback is part of a formal run.

- `packages/model/src/model/`: Model Core streaming model, encoders, attention,
  speech head, action head, and recurrent state. Keep model math here; do not
  put orchestration here.
- `systems/training/src/training/`: the shared optimizer/recipe dispatch,
  TBPTT, checkpoint lineage, metrics, and validation/test orchestration.
- `packages/data/src/data/`: synthetic data, WebDataset readers, trajectory
  schema, speech import, codec targets, and curation APIs.
- The legacy `src/latentloop/` package and its single-package CLI are removed.
  Formal imports and commands use `model`, `data`, `runtime`, `training`,
  `contracts`, `media`, `harness`, and `model-service` workspace boundaries.
- `packages/data/src/data/curation/`: source locks, manifests, synthesis,
  audits, Mimi checks, readiness, and encoded shard preparation.
- `systems/training/src/training/evaluation.py`: shared validation/test
  evaluation and lineage-bearing reports.
- `systems/training/src/training/checkpoint.py`: atomic checkpoint save/load
  and identity validation. Do not bypass it for normal training.
- `systems/training/src/training/tracking.py`: W&B Local/online/offline
  integration.
- `configs/`: model/data/training profiles, stage configs, and recipes.
- `scripts/`: thin, fail-fast wrappers around `uv run` commands and workers.
- `tools/`: external adapters and dataset utilities. Keep external model
  environments isolated in their existing subprojects.
- `tests/`: unit, contract, integration, checkpoint, curation, and training
  tests.
- `docs/`: architecture, local platform, data runbooks, and experiment notes.

## Architecture Contracts

Do not silently change these invariants:

- Stream clock: one unit is 80 ms, 24 kHz audio, and exactly one Mimi frame.
- Codec identity: `mimi-24khz-8x2048`, eight codebooks, vocabulary 2048,
  with the configured revision and weight SHA-256.
- Context: the formal Canary profile retains 30 seconds (`375` units) of bounded
  per-layer KV state. KV is bounded and oldest context is evicted according to
  the model implementation; do not introduce unbounded cache growth.
- State: recurrent KV, semantic memory slots, per-layer SlowMemory, audio cache, and speech-local state are
  carried across TBPTT chunks within an episode and detached between chunks.
  State resets at episode/session boundaries, not at every sampling window.
- Inputs: training examples use the single mixed microphone signal plus screen
  input. Do not add privileged source stems, playback-reference channels, or
  inference-only labels to the model input contract.
- Outputs: speech is direct codec-token prediction. TTS is a data-preparation
  adapter only; it is not a second runtime speech target. Actions remain an
  independent action head.
- Long-term memory has no independent target or auxiliary write loss. It is
  trained through future Speech/Action losses flowing from heads through
  `H_t -> Backbone -> C_t/SlowMemory`, while JEPA supervises prediction from `H_t`.
- Do not reintroduce balanced-window training or per-window state reset. If
  speech supervision is sparse, fix the dataset composition and report the
  resulting supervision density.

## Configurations and Recipes

Use the existing profiles rather than creating ad hoc command-line model
definitions:

- `configs/smoke.yaml`: fast synthetic development and tests.
- `configs/local-dev.yaml`: local synthetic GPU profile.
- `configs/canary.yaml`: real Canary profile.
- `configs/direct-speech-overfit.yaml`: deterministic speech gate.
- `configs/stages/*.yaml`: complete stage configurations inherited from a
  profile.
- `configs/recipes/*.yaml`: ordered stage composition and evaluation policy.

The Canary recipe must keep validation after each stage
and test after the final stage unless a documented experiment explicitly says
otherwise. Use fixed `max_updates` for reproducible runs. Configuration
overrides are for experiments and short regressions, not for hiding a changed
formal contract.

## Data and External Assets

The default data root is `~/latentloop-data/datasets`; generated datasets,
audio, model weights, sockets, checkpoints, and W&B state do not belong in the
Git repository.

Data preparation is fail-closed:

```bash
scripts/prepare-data.sh canary all
uv run data check-readiness --config configs/canary.yaml
```

Real Canary assets must have locked source versions, licenses, archive hashes,
manifests, audit reports, encoded shards, and Mimi reports. Never fabricate a
formal asset or silently fall back to a fixture. Fixture data is allowed only for explicit local tests and
must remain visibly marked as fixture.

Readiness must verify the configured train manifest and shards belong to the
selected dataset and that Mimi identity and manifest hashes match. Do not
weaken a gate to make a run pass. If a real asset is unavailable, report the
missing asset and keep the failure explicit.

## Runs, Checkpoints, and Lineage

Use an explicit run ID for anything that may be resumed or compared:

```bash
scripts/run-training.sh \
  --recipe configs/recipes/canary.yaml \
  --run-id canary-001 \
  --set training.max_updates=5 \
  --set tracking.mode=offline
```

Artifacts are isolated under:

```text
<experiment-root>/<recipe>/<run-id>/<stage>/
```

Reuse a run ID only when intentionally resuming the exact same stage. A
changed config, model shape, data manifest, codec revision, or codec weight
must use a new run ID. The recipe runner rejects incompatible checkpoints;
never work around this by deleting metadata or manually loading weights.

Formal Canary Pretrain/SFT/RL stages always use
`backbone_train_mode=all`. SFT receives the preceding Pretrain checkpoint and
RL receives the final SFT checkpoint as both its initial policy and frozen
reference. Frozen/selective modes remain available only for explicitly named
non-formal local experiments and must not appear in formal recipes.

The final Harness backend is an isolated QEMU/KVM computer. A per-session
overlay is restored from a task snapshot; SPICE/virtual audio provide screen
and microphone integration and a controlled input backend executes validated
`ControlSignal` events. A process-local fake backend is test-only.

## W&B Local and Experiment Tracking

W&B Local is the default tracking target at `http://127.0.0.1:8080`.

- Use `tracking.mode=online` only when the local server is healthy and the CLI
  credentials are configured.
- Use `tracking.mode=offline` for reproducible runs without the server.
- Use `tracking.mode=disabled` only for tests or deliberately untracked smoke
  runs.
- Do not put API keys, `.netrc`, database files, or W&B media in Git.
- Preserve config hash, data identity, codec identity, Git state, run ID,
  stage name, and parent checkpoint hash in tracking metadata.
- Metrics must be logged at the configured update cadence. Do not add a second
  logging loop in a stage script.

## Development Workflow

Before editing, inspect the relevant config, implementation, tests, and docs.
Keep a change scoped to the requested behavior. Do not revert unrelated user
changes or rewrite generated artifacts.

### Documentation Preservation

文档更新必须以现有文档为基线进行增量修订。不得无理由大面积删减文档，
也不得把中文文档整体改写成英文或改变其主要语言。应保留原有章节结构、
背景说明、运行手册和完整设计上下文，并逐节对齐最终优雅目标架构。只有
确实与最终方案冲突且已无用的历史内容才可以删除；删除时要保留必要的
设计脉络和迁移说明，不能用一份简化摘要替代完整文档。涉及架构变更时，
先刷新文档和测试设计，再开始代码实现。

文档职责必须保持清晰：`docs/realtime-multimodal-latent-loop.md` 是本项目的
顶层架构文档，必须与当前
实现、数据协议、训练目标、运行时状态和最终目标方案一致；Speech Head 的专项
协议见 `docs/direct-speech.md`，统一电脑操控输出空间的专项协议见
`docs/unified-action.md`。顶层架构文档不得停留在脱离代码的研究草案，且应保留
完整结构和中文表达。

三阶段训练的专项契约由 `docs/three-stage-training.md` 描述，Online RL 的
Online Recurrent PPO 算法、真实隔离环境和 reward 协议由
`docs/online-recurrent-ppo-training.md` 描述。它们是最终系统
架构的一部分，不写中间讨论过程、开发排期或阶段性妥协方案；顶层架构和本地平台
文档应引用并保持语义一致。

For Python changes, use `apply_patch`, preserve type hints and existing local
patterns, and keep comments limited to non-obvious invariants. Prefer
structured parsers and APIs over ad hoc string parsing. Use ASCII for new code
unless the surrounding document intentionally uses another character set.

For shell changes:

- start with `set -euo pipefail`;
- quote paths and variables;
- fail on missing required arguments and external assets;
- keep wrappers thin and delegate behavior to Python modules or existing
  worker scripts;
- ensure cleanup traps stop workers on success, failure, and interruption.

Do not add dependencies without updating `pyproject.toml`/the relevant
subproject lockfile and explaining why the existing stack is insufficient.

## Verification Commands

Run the narrowest relevant tests while iterating, then run the full gate before
finishing:

```bash
uv run pytest -q
uv run ruff check src tests tools/curation
uv run python -m compileall -q src tools/curation
bash -n scripts/*.sh scripts/lib/*.sh
git diff --check
```

For configuration or data-contract changes also run:

```bash
uv run data inspect-model --config configs/smoke.yaml
uv run data check-readiness --config configs/canary.yaml
```

For a real-data end-to-end regression, use a unique run ID and a small update
count. The validation/test pass may take several minutes because it replays
the full split on the GPU; do not mistake a slow evaluation for a hung process
without checking GPU/process activity.

Expected real Canary artifacts include a passed readiness report, processed
train/validation/test shards, a checkpoint, and validation/test reports with
non-zero episode and speech-frame counts. Report runtime, update count,
consumed units, supervision density, peak memory, and W&B mode when summarizing
the run.

## Tests and Regression Expectations

Add or update tests with behavior changes:

- config validation for new fields and invariants;
- recipe tests for stage chaining, run isolation, resume, and incompatible
  checkpoint rejection;
- readiness tests for missing assets, wrong dataset paths, codec mismatch,
  failed Mimi segments, and hash mismatch;
- checkpoint tests for atomic save/load and identity checks;
- model/training tests for state continuity, TBPTT detach behavior, metrics,
  and gradient accumulation;
- CLI/script tests for public command names and failure modes.

Do not assert that a short Canary run has converged. A short Canary run proves
the data, codec, training, checkpoint, and evaluation pipeline is executable;
quality claims require a documented update budget and meaningful metrics.

## Git and Completion Checklist

Before declaring work complete:

1. Confirm the requested behavior is implemented through the shared code path.
2. Confirm no stage-specific duplicate loop or obsolete command was added.
3. Add/update focused tests and run the full verification commands.
4. Check that no data, checkpoint, secret, socket, W&B database, or generated
   cache is staged.
5. Review `git diff --check` and the final status.
6. Summarize changed files, verification results, and any external assets or
   long-running checks that were not available.

Commits should be focused and use an imperative subject, for example:
`Unify training stages through shared recipe path`. Do not amend or rewrite
existing commits unless the user explicitly asks for it.
