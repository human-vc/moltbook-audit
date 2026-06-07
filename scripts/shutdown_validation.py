from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime

import numpy as np
import polars as pl

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_rigor as R

BREACH = datetime(2026, 1, 31, 17, 35, 0)
RESTART = datetime(2026, 2, 3, 13, 25, 0)


def _confusion(label_auto, pred_auto):
    label_auto = np.asarray(label_auto, bool)
    pred_auto = np.asarray(pred_auto, bool)
    tp = int((label_auto & pred_auto).sum())
    fp = int((~label_auto & pred_auto).sum())
    fn = int((label_auto & ~pred_auto).sum())
    tn = int((~label_auto & ~pred_auto).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    acc = (tp + tn) / max(1, tp + fp + fn + tn)
    tpr = rec
    tnr = tn / (tn + fp) if tn + fp else 0.0
    bal_acc = 0.5 * (tpr + tnr)
    denom = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = ((tp * tn - fp * fn) / denom) if denom else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": round(prec, 3), "recall": round(rec, 3), "f1": round(f1, 3),
            "accuracy": round(acc, 3), "balanced_accuracy": round(bal_acc, 3),
            "mcc": round(float(mcc), 3)}


def _auc(label_auto, cov):
    label_auto = np.asarray(label_auto, bool)
    score = -np.asarray(cov, float)
    pos = score[label_auto]
    neg = score[~label_auto]
    if not len(pos) or not len(neg):
        return None
    wins = sum((pos[:, None] > neg[None, :]).sum() for _ in [0])
    ties = (pos[:, None] == neg[None, :]).sum()
    return round(float((wins + 0.5 * ties) / (len(pos) * len(neg))), 3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--start", default="2026-01-27")
    ap.add_argument("--end", default="2026-02-20")
    ap.add_argument("--cov-window", choices=["full", "prebreach"], default="full")
    ap.add_argument("--cov-threshold", type=float, default=0.75)
    ap.add_argument("--return-hours", type=float, default=6.0)
    ap.add_argument("--min-pre-events", type=int, default=5)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--out", default="results/shutdown_validation.json")
    args = ap.parse_args()

    posts, comments = R.prepare(args.data_dir, args.start, args.end, None, None)

    if args.cov_window == "prebreach":
        cov_df = R.autonomy_cov(posts.filter(pl.col("created_at") < BREACH),
                                comments.filter(pl.col("created_at") < BREACH))
    else:
        cov_df = R.autonomy_cov(posts, comments)

    ret = (posts.filter(pl.col("created_at") >= RESTART)
           .group_by("agent_id").agg(first_return=pl.col("created_at").min()))
    ret = ret.with_columns(
        ((pl.col("first_return") - pl.lit(RESTART)).dt.total_seconds() / 3600.0).alias("hours_to_return"))

    cov_pl = pl.from_pandas(cov_df[["agent_id", "cov", "n_events"]])
    m = cov_pl.join(ret.select(["agent_id", "hours_to_return"]), on="agent_id", how="inner")
    m = m.filter(pl.col("n_events") >= args.min_pre_events)

    cov = m["cov"].to_numpy()
    htr = m["hours_to_return"].to_numpy()
    label_auto = htr > args.return_hours
    pred_auto = cov <= args.cov_threshold

    base = {
        "shutdown_window_utc": [BREACH.isoformat(), RESTART.isoformat()],
        "label_rule": f"autonomous = first post-restart later than {args.return_hours}h (Li-style; early returners are human-steered)",
        "pred_rule": f"autonomous = {args.cov_window}-window CoV <= {args.cov_threshold}",
        "leakage_note": ("full-window CoV mildly overlaps the post-restart label period; "
                         "pre-breach CoV is leakage-free but noisy (~8 events/agent)"),
        "n_eligible": int(m.height),
        "n_labeled_autonomous": int(label_auto.sum()),
        "n_labeled_human": int((~label_auto).sum()),
        "frac_labeled_autonomous": round(float(label_auto.mean()), 3),
        "confusion": _confusion(label_auto, pred_auto),
        "roc_auc_cov_vs_shutdown_label": _auc(label_auto, cov),
        "note": ("the 0.75 CoV cutoff is THIS paper's, not Li's (Li uses 0.5/1.0); "
                 "shutdown-return labels are a behavioral proxy with a fuzzy boundary"),
    }

    rng = np.random.RandomState(R.SEED)
    n = len(cov)
    boot = []
    for _ in range(args.n_boot):
        idx = rng.randint(0, n, n)
        boot.append(_confusion(label_auto[idx], pred_auto[idx])["f1"])
    base["f1_ci95"] = [round(float(np.percentile(boot, 2.5)), 3),
                       round(float(np.percentile(boot, 97.5)), 3)]

    sweeps = {}
    for rh in (3.0, 6.0, 12.0):
        la = htr > rh
        sweeps[f"return_hours={rh}"] = {
            "n_labeled_autonomous": int(la.sum()),
            "confusion": _confusion(la, pred_auto),
            "roc_auc": _auc(la, cov)}
    base["return_window_sensitivity"] = sweeps

    cuts = {}
    for c in (0.3, 0.5, 0.75, 1.0):
        cuts[f"cov<={c}"] = _confusion(label_auto, cov <= c)
    base["cov_threshold_sensitivity"] = cuts

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(base, f, indent=2, default=str)
    print(json.dumps(base, indent=2, default=str))


if __name__ == "__main__":
    main()
