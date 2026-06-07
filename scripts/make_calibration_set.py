from __future__ import annotations

import argparse
import os

import numpy as np
import polars as pl

SEED = 42
DIMS = ["correctness", "completeness", "justification", "actionability", "clarity", "overall"]


def stratified(df, key, n, seed):
    if df.height <= n:
        return df
    rng = np.random.RandomState(seed)
    labels = df[key].to_numpy()
    total = len(labels)
    picks = []
    for lab in np.unique(labels):
        idx = np.where(labels == lab)[0]
        take = min(len(idx), max(1, int(round(n * len(idx) / total))))
        picks.append(rng.choice(idx, size=take, replace=False))
    sel = np.sort(np.concatenate(picks))
    return df[sel.tolist()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", default="results/judge_threads.parquet")
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--out", default=os.path.expanduser("~/Downloads/moltbook_coding_sheet.csv"))
    args = ap.parse_args()

    df = pl.read_parquet(args.threads)
    df = df.with_columns(
        (pl.col("contrib_bucket") + "_" + pl.col("has_code").cast(pl.Utf8)).alias("_strata"))
    sub = stratified(df, "_strata", args.n, SEED).sort(["contrib_bucket", "post_id"])
    sub = sub.with_columns(pl.int_range(1, sub.height + 1).alias("row_id"))

    for d in DIMS:
        sub = sub.with_columns(pl.lit("").alias(f"{d}_human"))
    sub = sub.with_columns(pl.lit("").alias("notes"))

    cols = (["row_id", "post_id", "n_contributors", "n_comments", "has_code", "thread_text"]
            + [f"{d}_human" for d in DIMS] + ["notes"])
    out = sub.select(cols)
    out.write_csv(args.out)
    print(f"wrote {out.height} rows to {args.out}")
    print("strata coverage:",
          dict(zip(*[list(x) for x in np.unique(sub["_strata"].to_numpy(), return_counts=True)])))


if __name__ == "__main__":
    main()
