from __future__ import annotations

import glob
import json
import sys
from datetime import datetime

import numpy as np
import polars as pl
from scipy.stats import rankdata, spearmanr

sys.path.insert(0, "scripts")
import run_rigor as R
from salvage_classifier import REGEX

DD = "moltbook-observatory-archive/moltbook-observatory-archive/data"
RESTART = datetime(2026, 2, 3, 13, 25)


def auc(score, label):
    s = np.asarray(score, float); y = np.asarray(label, bool)
    n1 = int(y.sum()); n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return None
    r = rankdata(s)
    return float((r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def main():
    posts, comments = R.prepare(DD, "2026-01-28", "2026-02-20", None, None)
    cov = pl.from_pandas(R.autonomy_cov(posts, comments)[["agent_id", "cov", "n_events"]])

    ptext = posts.select(["agent_id", (pl.col("title").fill_null("") + " " + pl.col("content").fill_null("")).alias("t")])
    ctext = comments.select(["agent_id", pl.col("content").fill_null("").alias("t")])
    disc = set(pl.concat([ptext, ctext]).filter(pl.col("t").str.contains(REGEX))["agent_id"].unique().to_list())

    ag = pl.scan_parquet(sorted(glob.glob(DD + "/agents/*.parquet"))).select(
        ["id", "is_claimed", "owner_x_handle"]).collect().with_columns(pl.col("id").cast(pl.Utf8).alias("agent_id"))

    # shutdown return time
    ret = (posts.filter(pl.col("created_at") >= RESTART).group_by("agent_id")
           .agg(first_ret=pl.col("created_at").min())
           .with_columns(((pl.col("first_ret") - pl.lit(RESTART)).dt.total_seconds() / 3600).alias("hours_to_return")))

    d = (cov.join(ag, on="agent_id", how="left").join(ret.select(["agent_id", "hours_to_return"]), on="agent_id", how="left")
         .with_columns([
             pl.col("agent_id").is_in(list(disc)).alias("self_ai"),
             (pl.col("owner_x_handle").is_not_null() & (pl.col("owner_x_handle").cast(pl.Utf8).str.len_chars() > 0)).alias("has_x"),
             pl.col("is_claimed").fill_null(False),
         ])).to_pandas()

    covv = d["cov"].to_numpy(); nev = d["n_events"].to_numpy()
    self_ai = d["self_ai"].to_numpy(); has_x = d["has_x"].to_numpy(); claimed = d["is_claimed"].to_numpy().astype(bool)
    out = {}

    # 1. convergent validity: does high cov mark the non-human side across THREE independent labels?
    out["mean_cov"] = {"self_disclosed_AI": round(float(covv[self_ai].mean()), 3),
                       "not_self_AI": round(float(covv[~self_ai].mean()), 3),
                       "unclaimed": round(float(covv[~claimed].mean()), 3), "claimed": round(float(covv[claimed].mean()), 3),
                       "no_x_handle": round(float(covv[~has_x].mean()), 3), "x_linked": round(float(covv[has_x].mean()), 3)}
    out["AUC_highcov_predicts"] = {"self_AI(auto)": round(auc(covv, self_ai), 3),
                                   "unclaimed(non-human)": round(auc(covv, ~claimed), 3),
                                   "no_x(non-human)": round(auc(covv, ~has_x), 3)}

    # 2. VOLUME CONFOUND: cov vs n_events, and cov->autonomy AUC WITHIN n_events quartiles (clean 2-class)
    out["spearman_cov_vs_nevents"] = round(float(spearmanr(covv, nev).statistic), 3)
    mask = self_ai | (has_x & ~self_ai)
    cm, ym, nm = covv[mask], self_ai[mask], nev[mask]
    q = np.quantile(nm, [0.25, 0.5, 0.75])
    out["cov_AUC_within_nevent_quartiles"] = {}
    edges = [(-1, q[0]), (q[0], q[1]), (q[1], q[2]), (q[2], 1e18)]
    for i, (lo, hi) in enumerate(edges):
        sel = (nm > lo) & (nm <= hi)
        out["cov_AUC_within_nevent_quartiles"][f"Q{i+1}_n<= {hi:.0f}"] = {
            "n": int(sel.sum()), "auc_cov_vs_selfAI": round(auc(cm[sel], ym[sel]), 3) if sel.sum() > 20 else None,
            "mean_cov_auto": round(float(cm[sel & (ym == 1)].mean()), 3) if (sel & (ym == 1)).sum() else None,
            "mean_cov_human": round(float(cm[sel & (ym == 0)].mean()), 3) if (sel & (ym == 0)).sum() else None}

    # 3. SHUTDOWN RECONCILIATION: Li assumed early-return=human. Do OUR labels agree?
    rr = d.dropna(subset=["hours_to_return"])
    rsa = rr["self_ai"].to_numpy(); rhx = rr["has_x"].to_numpy(); rh = rr["hours_to_return"].to_numpy(); rc = rr["cov"].to_numpy()
    out["shutdown_reconcile"] = {
        "median_hours_return_self_AI": round(float(np.median(rh[rsa])), 2) if rsa.sum() else None,
        "median_hours_return_x_linked": round(float(np.median(rh[rhx & ~rsa])), 2) if (rhx & ~rsa).sum() else None,
        "spearman_cov_vs_return_hours": round(float(spearmanr(rc, rh).statistic), 3),
        "note": "if self-AI return LATER than humans => Li's early=human holds AND high-cov=auto holds (consistent); "
                "spearman(cov,return)>0 means high-cov agents return later (autonomous resume on own clock)"}

    with open("results/verify_polarity.json", "w") as f:
        json.dump(out, f, indent=2)
    print("\n=== POLARITY VERIFICATION ===")
    print("mean CoV by label:", out["mean_cov"])
    print("AUC high-cov predicts non-human side:", out["AUC_highcov_predicts"])
    print(f"\nVOLUME CONFOUND: spearman(cov, n_events) = {out['spearman_cov_vs_nevents']}")
    print("cov->autonomous AUC within n_events quartiles (de-confounded):")
    for k, v in out["cov_AUC_within_nevent_quartiles"].items():
        print(f"  {k}: AUC={v['auc_cov_vs_selfAI']}  cov auto={v['mean_cov_auto']} vs human={v['mean_cov_human']}  (n={v['n']})")
    print("\nSHUTDOWN RECONCILE:", out["shutdown_reconcile"])


if __name__ == "__main__":
    main()
