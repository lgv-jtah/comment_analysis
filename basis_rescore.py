"""Offline basis re-score (usage: uv run python basis_rescore.py) of saved policy-v2 predictions. No inference, no network.

Saved rows do not record the revised scope, so the engine rule cannot be replayed
exactly. Two reconstruction tiers, reported separately:
  exact:    the comment already has a general_approval row, so general support and
            its evidence are observed; the engine rule is applied verbatim.
  inferred: no general_approval row, >=2 targets, every target explicit_target /
            in_favor with the same nonempty evidence; general support with that
            shared evidence is assumed (a blanket approval quoted per target).
The basis relabelling alone never changes labels, so headline accuracy is invariant.

Option C (2026-10-06 decision, docs/stance_policy_v2.md) then sets every
general_approval row's stance to neutral, as the current engines do. Reported
for the exact tier alone (lower bound of the change) and exact+inferred (upper
bound). The rule was chosen after inspecting this cohort, so these are
development-set simulations, not held-out results.
"""
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from evaluator_v2 import accepted_accuracy_by_basis, evaluate  # noqa: E402
from policy import GENERAL_APPROVAL_STANCE, restates_general_approval  # noqa: E402

EXP = ROOT / "dataset/experiments"
RUNS = {
    "gpt": ("model_comparison_checkpointed_20260929T064921105737Z_gptoss-120b_full.parquet",
            "model_comparison_checkpointed_20260929T064921105737Z_gptoss-120b_full.expected.parquet"),
    "qwen_partial": ("model_comparison_checkpointed_20260929T064921105737Z_qwen35-397b_full.partial.parquet",
                     "model_comparison_checkpointed_20260929T064921105737Z_qwen35-397b_full.expected.parquet"),
    "nemo": ("screening_fold0_evidence_ids_20260928T142234517635Z_live.parquet",
             "screening_fold0_evidence_ids_20260928T142234517635Z_live.expected.parquet"),
}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def relabel(pred):
    pred = pred.copy()
    pred["rescore_tier"] = None
    for _, group in pred.groupby(["contribution_id", "comment_text"], sort=False):
        general = group[group.basis == "general_approval"]
        if len(general):
            tier, evidence = "exact", list(general.iloc[0].comment_evidence)
        else:
            directional = group[(group.basis == "explicit_target") & (group.proposed_stance == "in_favor")]
            shared = {tuple(e) for e in directional.comment_evidence}
            if len(group) < 2 or len(directional) != len(group) or len(shared) != 1 or not next(iter(shared)):
                continue
            tier, evidence = "inferred", list(next(iter(shared)))
        for index, row in group.iterrows():
            if (row.basis == "explicit_target" and row.proposed_stance == "in_favor"
                    and restates_general_approval(list(row.comment_evidence), evidence)):
                pred.at[index, "basis"] = "general_approval"
                pred.at[index, "comment_evidence"] = evidence
                pred.at[index, "rescore_tier"] = tier
    return pred


def option_c(pred, tiers=("exact", "inferred")):
    """Apply option C to relabelled rows: general approval is neutral per key message."""
    pred = pred.copy()
    keep = pred.basis == "general_approval"
    keep &= pred.rescore_tier.isna() | pred.rescore_tier.isin(tiers)
    revert = (pred.basis == "general_approval") & ~keep
    pred.loc[revert, "basis"] = "explicit_target"
    for column in ("proposed_stance", "final_stance"):
        pred.loc[keep & pred[column].notna(), column] = GENERAL_APPROVAL_STANCE.value
    return pred


def by_basis(pred, ref, expected, tiers):
    return {k: {"correct": v.correct, "n": v.n, "accuracy": round(v.accuracy, 4)}
            for k, v in sorted(accepted_accuracy_by_basis(pred, ref, expected, tiers=tiers).items())}


def main():
    ref_path = ROOT / "dataset/pairs.parquet"
    ref = pd.read_parquet(ref_path)
    inputs = {"reference": sha(ref_path), "script": sha(Path(__file__).resolve())}
    out = {"created_utc": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ"),
           "analysis_only": True, "new_inference_requests": 0, "method": __doc__.strip(), "runs": {}}
    for name, (pred_file, exp_file) in RUNS.items():
        inputs[pred_file], inputs[exp_file] = sha(EXP / pred_file), sha(EXP / exp_file)
        pred = pd.read_parquet(EXP / pred_file)
        expected = pd.read_parquet(EXP / exp_file)
        new = relabel(pred)
        assert (new.final_stance.fillna("~") == pred.final_stance.fillna("~")).all()
        assert (new.status == pred.status).all()
        run = {}
        for label, tiers in (("primary_A", ("A",)), ("primary_AB", ("A", "B"))):
            before = evaluate(pred, ref, expected, tiers=tiers)
            after = evaluate(new, ref, expected, tiers=tiers)
            assert before.accepted_correct == after.accepted_correct
            run[label] = {
                "accepted_correct": after.accepted_correct, "accepted_pairs": after.accepted_pairs,
                "eligible_pairs": after.eligible_pairs,
                "by_basis_before": by_basis(pred, ref, expected, tiers),
                "by_basis_after": by_basis(new, ref, expected, tiers),
                "option_c": {},
            }
            for bound, relabel_tiers in (("exact_only", ("exact",)), ("exact_and_inferred", ("exact", "inferred"))):
                simulated = option_c(new, relabel_tiers)
                metrics = evaluate(simulated, ref, expected, tiers=tiers)
                run[label]["option_c"][bound] = {
                    "accepted_correct": metrics.accepted_correct, "accepted_pairs": metrics.accepted_pairs,
                    "eligible_pairs": metrics.eligible_pairs,
                    "correct_automatic_yield": round(metrics.accepted_correct / metrics.eligible_pairs, 4),
                    "by_basis": by_basis(simulated, ref, expected, tiers),
                }
        changed = new[new.rescore_tier.notna()]
        run["relabelled_rows"] = {"total": len(changed), **changed.rescore_tier.value_counts().to_dict()}
        run["relabelled_comments"] = int(changed.groupby(["contribution_id", "comment_text"]).ngroups)
        out["runs"][name] = run
    out["input_sha256"] = inputs
    target = EXP / f"basis_rescore_{out['created_utc']}.safe_summary.json"
    target.write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n")
    print(target)
    for name, run in out["runs"].items():
        ab = run["primary_AB"]
        print(f"\n== {name}: accepted {ab['accepted_correct']}/{ab['accepted_pairs']} (eligible {ab['eligible_pairs']}), "
              f"relabelled {run['relabelled_rows']} on {run['relabelled_comments']} comments")
        for key in sorted(set(ab["by_basis_before"]) | set(ab["by_basis_after"])):
            b, a = ab["by_basis_before"].get(key), ab["by_basis_after"].get(key)
            fmt = lambda m: f"{m['correct']}/{m['n']} ({m['accuracy']:.3f})" if m else "-"
            print(f"  {key:20s} before {fmt(b):18s} after {fmt(a)}")
        for bound, metrics in ab["option_c"].items():
            print(f"  option C ({bound}): {metrics['accepted_correct']}/{metrics['eligible_pairs']} "
                  f"correct automatic ({metrics['correct_automatic_yield']:.3f})")


if __name__ == "__main__":
    main()
