from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import date, timedelta

import polars as pl

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_rigor as R


def slice_stats(posts, comments, null_iters, leiden_null, shuffle_iters, cov_threshold, n_jobs):
    g = R.build_graph(R.build_edges(posts, comments))
    nulls = R.task1_nulls(g, null_iters, n_jobs)
    recip = nulls.get("reciprocity", {})
    ratio = (recip["observed"] / recip["null_mean"]) if recip.get("null_mean") else None
    leiden = R.task1_leiden(g, n_null=leiden_null)
    casc = R.task2(posts, shuffle_iters, gof_iters=0)
    verdict = casc.get("powerlaw", {}).get("verdict", casc.get("result"))
    cov = R.autonomy_cov(posts, comments)
    human_share = None
    if hasattr(cov, "empty") and not cov.empty and comments.height:
        human_ids = set(cov.loc[cov["cov"] > cov_threshold, "agent_id"])
        human_share = float(comments.filter(pl.col("agent_id").is_in(human_ids)).height / comments.height)
    return {
        "n_nodes": g.vcount(), "n_edges": g.ecount(),
        "reciprocity_obs": recip.get("observed"),
        "reciprocity_null_mean": recip.get("null_mean"),
        "reciprocity_ratio": round(ratio, 2) if ratio else None,
        "reciprocity_emp_p": recip.get("emp_p"),
        "modularity": leiden.get("modularity"),
        "modularity_z": leiden.get("modularity_vs_null", {}).get("z"),
        "n_cascades": casc.get("n_cascades"),
        "cascade_verdict": verdict,
        "human_steered_comment_share": round(human_share, 3) if human_share is not None else None,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--start", default="2026-01-28")
    ap.add_argument("--end", default="2026-02-20")
    ap.add_argument("--n-windows", type=int, default=3)
    ap.add_argument("--exclude-dates", default=None)
    ap.add_argument("--null-iters", type=int, default=30)
    ap.add_argument("--leiden-null", type=int, default=15)
    ap.add_argument("--shuffle-iters", type=int, default=15)
    ap.add_argument("--cov-threshold", type=float, default=0.75)
    ap.add_argument("--n-jobs", type=int, default=-1)
    ap.add_argument("--no-full", action="store_true")
    ap.add_argument("--out", default="results/subwindow_invariance.json")
    args = ap.parse_args()

    excl = [d.strip() for d in args.exclude_dates.split(",")] if args.exclude_dates else None
    lo, hi = date.fromisoformat(args.start), date.fromisoformat(args.end)
    span = (hi - lo).days + 1
    step = span // args.n_windows
    windows = []
    for w in range(args.n_windows):
        s = lo + timedelta(days=w * step)
        e = (lo + timedelta(days=(w + 1) * step - 1)) if w < args.n_windows - 1 else hi
        windows.append((f"week{w + 1}", s.isoformat(), e.isoformat()))
    if not args.no_full:
        windows.append(("full", args.start, args.end))

    rows = []
    for name, s, e in windows:
        R.log.info("=== subwindow %s %s..%s ===", name, s, e)
        posts, comments = R.prepare(args.data_dir, s, e, None, excl)
        st = R._safe(lambda p=posts, c=comments: slice_stats(
            p, c, args.null_iters, args.leiden_null, args.shuffle_iters, args.cov_threshold, args.n_jobs))
        st.update({"window": name, "start": s, "end": e})
        rows.append(st)

    out = {"framing": "robustness / window-invariance, not a temporal-dynamics model", "windows": rows}
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2, default=str)

    cols = ["window", "n_nodes", "n_edges", "reciprocity_ratio", "modularity",
            "modularity_z", "cascade_verdict", "human_steered_comment_share"]
    print("\n| " + " | ".join(cols) + " |")
    print("|" + "|".join(["---"] * len(cols)) + "|")
    for r in rows:
        print("| " + " | ".join(str(r.get(c)) for c in cols) + " |")


if __name__ == "__main__":
    main()
