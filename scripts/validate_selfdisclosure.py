from __future__ import annotations

import json
import sys

import numpy as np
import polars as pl

sys.path.insert(0, "scripts")
import run_rigor as R
from salvage_classifier import REGEX, DD


def main():
    posts, comments = R.prepare(DD, "2026-01-28", "2026-02-20", None, None)
    ptext = posts.select(["agent_id", (pl.col("title").fill_null("") + " " + pl.col("content").fill_null("")).alias("t")])
    ctext = comments.select(["agent_id", pl.col("content").fill_null("").alias("t")])
    allm = pl.concat([ptext, ctext])
    matched = allm.filter(pl.col("t").str.contains(REGEX))
    per_agent = matched.group_by("agent_id").agg(pl.col("t").first().alias("t"))
    rng = np.random.default_rng(42)
    idx = rng.choice(per_agent.height, size=min(30, per_agent.height), replace=False)
    samp = per_agent[sorted(idx.tolist())]
    rows = []
    print(f"=== {per_agent.height} flagged agents; sampling 30 for precision ===\n")
    for i, (aid, t) in enumerate(zip(samp["agent_id"], samp["t"]), 1):
        snip = " ".join(t.split())[:380]
        print(f"[{i}] {snip}\n")
        rows.append({"n": i, "agent_id": aid, "snippet": snip})
    with open("results/selfdisclosure_sample.json", "w") as f:
        json.dump(rows, f, indent=2)


if __name__ == "__main__":
    main()
