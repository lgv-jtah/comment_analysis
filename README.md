# Stance Detection App

See [Stance Policy v2](docs/stance_policy_v2.md) for classification and review rules.
The opt-in [Revisable Attribution Experiment](docs/revisable_attribution_experiment.md) documents offline preparation, IONOS-only execution, durable per-comment checkpoint/resume, and full expected-cohort scoring without changing the default engine.
Live `benchmark` CLI runs checkpoint by default; `--resume` preserves the run contract, `--recover-stale-lock` acknowledges an interrupted session without bypassing a live lock, and `--resume --evaluate-partial` scores saved results without inference.
There is no stage cache or recovery of old runs without checkpoints; omitted thinking overrides leave reasoning at the backend default, not explicitly disabled.
The 2026-09-29 checkpointed model comparison (GPT-OSS-120B complete, Qwen3.5-397B stopped by provider errors) ran under the pre-2026-10-06 general-approval rule with a temporary helper that no longer exists; its reservations cannot be resumed.
Existing report numbers are provisional and fold 0 is development data.
The single held-out run on folds 1-4 is recorded in the [experiment doc](docs/revisable_attribution_experiment.md#held-out-run-on-folds-1-4-2026-10-06); it does not reach the proposed 80% operating target, and its legacy comparisons are verified-by-construction rather than independent baselines.
