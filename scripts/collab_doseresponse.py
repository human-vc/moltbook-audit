from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import polars as pl

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_rigor as R


def thread_table(data_dir, start, end, exclude_dates, max_chars):
    posts, comments = R.prepare(data_dir, start, end, None, exclude_dates)
    pmeta = posts.select(["id", "agent_id", "submolt", "title", "content"]).rename(
        {"id": "post_id", "agent_id": "author", "content": "post_body"})
    th = (comments.group_by("post_id").agg(
            n_comments=pl.len(),
            n_commenters=pl.col("agent_id").n_unique(),
            commenters=pl.col("agent_id").unique(),
            bodies=pl.col("content").fill_null("").sort_by("created_at"),
            tmin=pl.col("created_at").min(), tmax=pl.col("created_at").max())
          .join(pmeta, on="post_id", how="inner")
          .filter(pl.col("n_comments") >= 3))
    th = th.with_columns(
        ((pl.col("tmax") - pl.col("tmin")).dt.total_seconds() / 60).alias("duration_min"),
        (pl.col("n_commenters")
         + (~pl.col("commenters").list.contains(pl.col("author"))).cast(pl.Int64)).alias("n_contributors"),
        pl.concat_str([pl.col("title").fill_null(""), pl.col("post_body").fill_null(""),
                       pl.col("bodies").list.join("\n")], separator="\n").str.slice(0, max_chars).alias("text"))
    t = pl.col("text").str.to_lowercase()
    th = th.with_columns(
        t.str.contains("|".join(R.TECHNICAL_KEYWORDS)).alias("is_technical"),
        t.str.contains(r"```|`[^`]+`").cast(pl.Float64).alias("_code"),
        t.str.contains(r"test|assert|expect").cast(pl.Float64).alias("_test"),
        t.str.contains(r"\b(works|fixed|solved|thanks|got it|resolved)\b").cast(pl.Float64).alias("_resolved"),
        pl.col("text").str.len_chars().alias("post_len"))
    th = th.filter(pl.col("is_technical")).with_columns(
        (0.4 * pl.col("_code") + 0.3 * pl.col("_test") + 0.3 * pl.col("_resolved")).alias("quality_heuristic"),
        pl.col("submolt").fill_null("unknown"))
    return th


def fit(df, dv):
    import statsmodels.api as sm
    import statsmodels.formula.api as smf
    from scipy.stats import spearmanr, rankdata, tiecorrect, norm
    d = df.to_pandas()
    d["log_nc"] = np.log(d["n_comments"].clip(lower=1))
    d["log_dur"] = np.log(d["duration_min"].clip(lower=1e-6) + 1e-6)
    d = d.rename(columns={dv: "quality"})
    d = d[np.isfinite(d["quality"])]

    out = {"n_threads": int(len(d)), "dv": dv,
           "n_contributors_range": [int(d["n_contributors"].min()), int(d["n_contributors"].max())]}

    ols = smf.ols("quality ~ n_contributors + log_nc + log_dur", data=d).fit(
        cov_type="cluster", cov_kwds={"groups": d["submolt"]})
    out["ols_cluster"] = {"slope_n_contributors": float(ols.params["n_contributors"]),
                          "se": float(ols.bse["n_contributors"]),
                          "p": float(ols.pvalues["n_contributors"]),
                          "ci95": [float(ols.conf_int().loc["n_contributors", 0]),
                                   float(ols.conf_int().loc["n_contributors", 1])],
                          "n_clusters": int(d["submolt"].nunique())}

    glm = smf.glm("quality ~ n_contributors + log_nc + log_dur", data=d,
                  family=sm.families.Binomial()).fit(cov_type="cluster", cov_kwds={"groups": d["submolt"]})
    out["fractional_binomial_glm"] = {"logit_coef_n_contributors": float(glm.params["n_contributors"]),
                                      "se": float(glm.bse["n_contributors"]),
                                      "p": float(glm.pvalues["n_contributors"])}

    rho, prho = spearmanr(d["n_contributors"], d["quality"])
    out["spearman"] = {"rho": float(rho), "p": float(prho)}

    vals = d["quality"].to_numpy(float)
    scores = d["n_contributors"].to_numpy(float)
    N = vals.size
    ranks = rankdata(vals)
    tc = tiecorrect(ranks)
    T = float(np.sum(scores * ranks))
    L = float(np.sum(scores))
    ET = 0.5 * (N + 1) * L
    VarT = (N + 1.0) / 12.0 * (N * float(np.sum(scores ** 2)) - L ** 2) * tc
    z = (T - ET) / np.sqrt(VarT) if VarT > 0 else 0.0
    out["cuzick_trend"] = {"z": float(z), "p": float(2 * norm.sf(abs(z)))}

    sign = "increases" if ols.params["n_contributors"] > 0 else "decreases"
    sig = ols.pvalues["n_contributors"] < 0.05
    out["reading"] = (f"quality {sign} with contributors (slope p={ols.pvalues['n_contributors']:.3g}); "
                      f"{'significant' if sig else 'no significant trend'} over {len(d)} threads")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--start", default="2026-02-02")
    ap.add_argument("--end", default="2026-05-28")
    ap.add_argument("--exclude-dates", default=None)
    ap.add_argument("--max-chars", type=int, default=6000)
    ap.add_argument("--scores", default=None, help="llm_judge parquet (post_id, overall) to use as DV")
    ap.add_argument("--out", default="results/collab_doseresponse.json")
    args = ap.parse_args()

    excl = [d.strip() for d in args.exclude_dates.split(",")] if args.exclude_dates else None
    th = thread_table(args.data_dir, args.start, args.end, excl, args.max_chars)

    res = {"window": [args.start, args.end], "n_threads_total": th.height}
    res["heuristic_quality"] = R._safe(lambda: fit(th, "quality_heuristic"))

    if args.scores and os.path.exists(args.scores):
        sc = pl.read_parquet(args.scores).select(["post_id", "overall"]).drop_nulls()
        merged = th.join(sc, on="post_id", how="inner").with_columns(
            ((pl.col("overall") - 1.0) / 4.0).clip(0.0, 1.0).alias("quality_judge"))
        res["judge_quality"] = R._safe(lambda: fit(merged, "quality_judge"))
        res["judge_quality"]["n_judged"] = merged.height

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(res, f, indent=2, default=str)
    print(json.dumps(res, indent=2, default=str))


if __name__ == "__main__":
    main()
