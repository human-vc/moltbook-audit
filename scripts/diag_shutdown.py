from __future__ import annotations

import os
import sys
from datetime import datetime

import numpy as np
import polars as pl

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_rigor as R

BREACH = datetime(2026, 1, 31, 17, 35, 0)
RESTART = datetime(2026, 2, 3, 13, 25, 0)


def auc(label_auto, score_low_is_auto):
    label_auto = np.asarray(label_auto, bool)
    s = -np.asarray(score_low_is_auto, float)
    pos, neg = s[label_auto], s[~label_auto]
    if not len(pos) or not len(neg):
        return None
    wins = (pos[:, None] > neg[None, :]).sum()
    ties = (pos[:, None] == neg[None, :]).sum()
    return round(float((wins + 0.5 * ties) / (len(pos) * len(neg))), 3)


def main():
    dd = sys.argv[1]
    posts, comments = R.prepare(dd, "2026-01-27", "2026-02-20", None, None)

    print("\n[1] HOURLY POST COUNTS around the shutdown (does the gap exist?)")
    h = (posts.filter((pl.col("created_at") >= datetime(2026, 1, 31, 0))
                      & (pl.col("created_at") < datetime(2026, 2, 4, 0)))
         .with_columns(pl.col("created_at").dt.truncate("1h").alias("hr"))
         .group_by("hr").agg(n=pl.len()).sort("hr"))
    for hr, n in zip(h["hr"].to_list(), h["n"].to_list()):
        bar = "#" * min(60, n // 50)
        flag = ""
        if BREACH <= hr < RESTART:
            flag = "  <-- OUTAGE WINDOW"
        print(f"  {hr}  {n:6d} {bar}{flag}")

    pre_p = posts.filter(pl.col("created_at") < BREACH)
    pre_c = comments.filter(pl.col("created_at") < BREACH)
    cov_pre = R.autonomy_cov(pre_p, pre_c)
    cov_full = R.autonomy_cov(posts, comments)

    ret = (posts.filter(pl.col("created_at") >= RESTART)
           .group_by("agent_id").agg(first_return=pl.col("created_at").min())
           .with_columns(((pl.col("first_return") - pl.lit(RESTART)).dt.total_seconds() / 3600.0)
                         .alias("htr")))

    for name, cov in (("pre-breach (~4d, posts-only)", cov_pre), ("full-window", cov_full)):
        cpl = pl.from_pandas(cov[["agent_id", "cov", "n_events"]])
        m = cpl.join(ret.select(["agent_id", "htr"]), on="agent_id", how="inner").filter(pl.col("n_events") >= 5)
        if not m.height:
            print(f"\n[2] {name}: no eligible agents")
            continue
        c = m["cov"].to_numpy(); htr = m["htr"].to_numpy(); ne = m["n_events"].to_numpy()
        late = htr > 6.0
        print(f"\n[2] {name}: eligible={m.height}  n_events(median={np.median(ne):.0f}, q25={np.percentile(ne,25):.0f})")
        print(f"    CoV  LATE-returners (labeled autonomous): median={np.median(c[late]):.3f} mean={c[late].mean():.3f}")
        print(f"    CoV  EARLY-returners (labeled human):     median={np.median(c[~late]):.3f} mean={c[~late].mean():.3f}")
        print(f"    AUC (low CoV predicts late/autonomous return): {auc(late, c)}")
        early = ~late
        hi_cov = c > 0.75
        print(f"    among EARLY returners: frac high-CoV(>0.75)={hi_cov[early].mean():.3f}  (Li reports ~0.877)")
        print(f"    overall frac high-CoV(>0.75)={hi_cov.mean():.3f}  (Li reports ~0.369)")


if __name__ == "__main__":
    main()
