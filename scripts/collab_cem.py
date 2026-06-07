from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import polars as pl

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_rigor as R
from collab_doseresponse import thread_table


def coarsen(d, spec):
    import pandas as pd
    out = pd.DataFrame(index=d.index)
    for col, rule in spec.items():
        if rule == "exact":
            out[col] = d[col].astype("category").cat.codes
        elif isinstance(rule, (list, tuple)):
            out[col] = pd.cut(d[col], bins=rule, include_lowest=True, labels=False, duplicates="drop")
        elif isinstance(rule, int):
            out[col] = pd.qcut(d[col], q=rule, labels=False, duplicates="drop")
        out[col] = out[col].fillna(-1).astype(int)
    return out


def cem_match(d, treat, spec):
    cx = coarsen(d, spec)
    strata = cx.astype(str).agg("|".join, axis=1)
    t = d[treat].astype(int).to_numpy()
    import pandas as pd
    g = pd.DataFrame({"s": strata.values, "t": t})
    c = g.groupby("s")["t"].agg(mT="sum", n="count")
    c["mC"] = c["n"] - c["mT"]
    keep = set(c[(c["mT"] > 0) & (c["mC"] > 0)].index)
    matched = strata.isin(keep).to_numpy()
    mT_tot = int(t[matched].sum())
    mC_tot = int(matched.sum() - mT_tot)
    sT, sC = c["mT"].to_dict(), c["mC"].to_dict()
    w = np.zeros(len(d))
    for i in np.where(matched)[0]:
        s = strata.iloc[i]
        w[i] = 1.0 if t[i] == 1 else (mC_tot / mT_tot) * (sT[s] / sC[s]) if mT_tot else 0.0
    return matched, w, {"n_matched_strata": len(keep), "matched_treat": mT_tot, "matched_control": mC_tot}


def l1_imbalance(d, treat, ref_spec):
    import pandas as pd
    cx = coarsen(d, ref_spec)
    cells = cx.astype(str).agg("|".join, axis=1)
    ct = pd.crosstab(cells, d[treat].astype(int))
    for col in (0, 1):
        if col not in ct:
            ct[col] = 0
    f = ct[1] / ct[1].sum() if ct[1].sum() else ct[1]
    g = ct[0] / ct[0].sum() if ct[0].sum() else ct[0]
    return float(0.5 * (f - g).abs().sum())


def run_cem(d, outcome):
    import statsmodels.api as sm
    spec = {"submolt": "exact",
            "n_comments": [-0.1, 2, 9, 29, np.inf],
            "duration_min": [0, 30, 1440, np.inf],
            "post_len": 4}
    ref = {"submolt": "exact", "n_comments": [-0.1, 5, np.inf], "duration_min": [0, 60, np.inf], "post_len": 3}
    n_treat, n_control = int(d["treat"].sum()), int((d["treat"] == 0).sum())
    l1_before = l1_imbalance(d, "treat", ref)
    matched, w, info = cem_match(d, "treat", spec)
    dm = d.copy()
    dm["w"], dm["matched"] = w, matched
    m = dm[dm["matched"]]
    l1_after = l1_imbalance(m, "treat", ref) if len(m) else None
    out = {"n_treat": n_treat, "n_control": n_control, **info,
           "control_match_rate": round(info["matched_control"] / n_control, 3) if n_control else None,
           "l1_before": round(l1_before, 3), "l1_after": round(l1_after, 3) if l1_after is not None else None}
    if info["matched_treat"] and info["matched_control"]:
        X = sm.add_constant(m[["treat"]].astype(float))
        wls = sm.WLS(m[outcome].astype(float), X, weights=m["w"]).fit(cov_type="HC1")
        out["att"] = {"estimate": float(wls.params["treat"]), "se": float(wls.bse["treat"]),
                      "p": float(wls.pvalues["treat"]),
                      "ci95": [float(wls.conf_int().loc["treat", 0]), float(wls.conf_int().loc["treat", 1])]}
    else:
        out["att"] = {"result": "no matched strata with both treated and control"}
    return out


def to_frame(th, dv):
    d = th.with_columns(
        ((pl.col("n_contributors") >= 3) & (pl.col("duration_min") >= 30) & (pl.col("n_comments") >= 5)).alias("_treat"),
        ((pl.col("n_contributors") == 1) & (pl.col("n_comments") >= 5)).alias("_control"))
    d = d.filter(pl.col("_treat") | pl.col("_control")).with_columns(
        pl.col("_treat").cast(pl.Int64).alias("treat"))
    cols = ["treat", dv, "submolt", "n_comments", "duration_min", "post_len"]
    return d.select(cols).to_pandas()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--start", default="2026-02-02")
    ap.add_argument("--end", default="2026-05-28")
    ap.add_argument("--exclude-dates", default=None)
    ap.add_argument("--max-chars", type=int, default=6000)
    ap.add_argument("--scores", default=None)
    ap.add_argument("--out", default="results/collab_cem.json")
    args = ap.parse_args()

    excl = [d.strip() for d in args.exclude_dates.split(",")] if args.exclude_dates else None
    th = thread_table(args.data_dir, args.start, args.end, excl, args.max_chars)
    if args.scores and os.path.exists(args.scores):
        sc = pl.read_parquet(args.scores).select(["post_id", "overall"]).drop_nulls()
        th = th.join(sc, on="post_id", how="left").with_columns(
            ((pl.col("overall") - 1.0) / 4.0).clip(0.0, 1.0).alias("quality_judge"))

    res = {"window": [args.start, args.end]}
    res["heuristic_quality"] = R._safe(lambda: run_cem(to_frame(th, "quality_heuristic"), "quality_heuristic"))
    if args.scores and "quality_judge" in th.columns:
        jt = to_frame(th.filter(pl.col("quality_judge").is_not_null()), "quality_judge")
        res["judge_quality"] = R._safe(lambda: run_cem(jt, "quality_judge"))

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(res, f, indent=2, default=str)
    print(json.dumps(res, indent=2, default=str))


if __name__ == "__main__":
    main()
