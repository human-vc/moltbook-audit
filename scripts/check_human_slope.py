from __future__ import annotations

import json
import sys

import numpy as np
import polars as pl
import statsmodels.formula.api as smf
from scipy import stats

hu = pl.read_csv("results/human_labels.csv").select(["post_id", "overall_human"])
js = pl.read_parquet("results/llm_judge_scores.parquet").select(
    ["post_id", "submolt", "n_contributors", "n_comments", "duration_min", "has_code", "overall"]
).rename({"overall": "overall_judge"})
th = pl.read_parquet("results/judge_threads.parquet").select(
    ["post_id", "thread_text"]
).with_columns(pl.col("thread_text").str.len_chars().alias("charlen")).drop("thread_text")

df = hu.join(js, on="post_id", how="inner").join(th, on="post_id", how="inner").drop_nulls(
    ["overall_human", "overall_judge", "n_contributors"]
)
d = df.to_pandas()
d["log_comments"] = np.log1p(d["n_comments"])
d["log_charlen"] = np.log1p(d["charlen"])
d["has_code"] = d["has_code"].astype(int)
n = len(d)


def slope(dv):
    sp = stats.spearmanr(d[dv], d["n_contributors"])
    m = smf.ols(f"{dv} ~ n_contributors + log_comments + duration_min + has_code + log_charlen", data=d).fit(
        cov_type="cluster", cov_kwds={"groups": d["submolt"]}
    )
    return {
        "spearman_rho": float(sp.statistic), "spearman_p": float(sp.pvalue),
        "ols_slope": float(m.params["n_contributors"]), "ols_se": float(m.bse["n_contributors"]),
        "ols_p": float(m.pvalues["n_contributors"]),
        "length_coef": float(m.params["log_charlen"]), "length_p": float(m.pvalues["log_charlen"]),
    }


# direct length-bias check: does judge track raw length more than humans do?
len_h = stats.spearmanr(d["overall_human"], d["charlen"])
len_j = stats.spearmanr(d["overall_judge"], d["charlen"])
contrib_len = stats.spearmanr(d["n_contributors"], d["charlen"])

out = {
    "n": n,
    "human": slope("overall_human"),
    "judge": slope("overall_judge"),
    "length_bias": {
        "human_overall_vs_charlen_rho": float(len_h.statistic),
        "judge_overall_vs_charlen_rho": float(len_j.statistic),
        "n_contributors_vs_charlen_rho": float(contrib_len.statistic),
    },
}
print(json.dumps(out, indent=2))
