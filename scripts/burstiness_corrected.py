from __future__ import annotations

import glob
import json
import sys

import numpy as np
import polars as pl
from scipy.stats import rankdata, spearmanr

sys.path.insert(0, "scripts")
import run_rigor as R
from salvage_classifier import REGEX, DD


def auc(score, label):
    s = np.asarray(score, float); y = np.asarray(label, bool)
    n1 = int(y.sum()); n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return None
    r = rankdata(s)
    return float((r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def kimjo_A(r, n):
    # A_n(r) = (sqrt(n+1) r - sqrt(n-1)) / ((sqrt(n+1)-2) r + sqrt(n-1)); valid 0<=r<=sqrt(n-1)
    sp, sm = np.sqrt(n + 1), np.sqrt(n - 1)
    return (sp * r - sm) / ((sp - 2) * r + sm)


def main():
    posts, comments = R.prepare(DD, "2026-01-28", "2026-02-20", None, None)
    ev = pl.concat([posts.select(["agent_id", "created_at"]),
                    comments.select(["agent_id", "created_at"])]).drop_nulls().sort(["agent_id", "created_at"])
    g = (ev.group_by("agent_id").agg(gaps=pl.col("created_at").diff().dt.total_seconds().drop_nulls())
         .filter(pl.col("gaps").list.len() >= 4))
    rows = []
    for aid, gaps in zip(g["agent_id"], g["gaps"]):
        a = np.array([x for x in gaps if x and x > 0], float)
        if len(a) < 4:
            continue
        m = a.mean()
        if m <= 0:
            continue
        r = a.std() / m
        n = len(a)
        rows.append((str(aid), float(r), int(n), float((r - 1) / (r + 1)), float(kimjo_A(r, n))))
    feat = pl.DataFrame(rows, schema=["agent_id", "cov", "n_gaps", "B_raw", "A_kimjo"], orient="row")

    ptext = posts.select(["agent_id", (pl.col("title").fill_null("") + " " + pl.col("content").fill_null("")).alias("t")])
    ctext = comments.select(["agent_id", pl.col("content").fill_null("").alias("t")])
    disc = set(pl.concat([ptext, ctext]).filter(pl.col("t").str.contains(REGEX))["agent_id"].unique().to_list())
    ag = pl.scan_parquet(sorted(glob.glob(DD + "/agents/*.parquet"))).select(["id", "owner_x_handle"]).collect().with_columns(
        pl.col("id").cast(pl.Utf8).alias("agent_id"))
    d = feat.join(ag, on="agent_id", how="left").with_columns([
        pl.col("agent_id").is_in(list(disc)).alias("self_ai"),
        (pl.col("owner_x_handle").is_not_null() & (pl.col("owner_x_handle").cast(pl.Utf8).str.len_chars() > 0)).alias("has_x"),
    ]).to_pandas()

    sub = d[d["self_ai"] | (d["has_x"] & ~d["self_ai"])]
    y = sub["self_ai"].to_numpy()
    out = {
        "n_agents": int(len(d)),
        "confound_check_spearman_vs_ngaps": {
            "raw_CoV": round(float(spearmanr(d["cov"], d["n_gaps"]).statistic), 3),
            "B_raw": round(float(spearmanr(d["B_raw"], d["n_gaps"]).statistic), 3),
            "A_kimjo_corrected": round(float(spearmanr(d["A_kimjo"], d["n_gaps"]).statistic), 3),
        },
        "clean2class": {"n_auto": int(y.sum()), "n_human": int((~y).sum())},
        "mean_by_class": {
            "CoV":     {"auto": round(float(sub[y == 1]["cov"].mean()), 3),     "human": round(float(sub[y == 0]["cov"].mean()), 3)},
            "B_raw":   {"auto": round(float(sub[y == 1]["B_raw"].mean()), 3),   "human": round(float(sub[y == 0]["B_raw"].mean()), 3)},
            "A_kimjo": {"auto": round(float(sub[y == 1]["A_kimjo"].mean()), 3), "human": round(float(sub[y == 0]["A_kimjo"].mean()), 3)},
        },
        "AUC_predicts_autonomous": {
            "raw_CoV": round(auc(sub["cov"].to_numpy(), y), 3),
            "B_raw": round(auc(sub["B_raw"].to_numpy(), y), 3),
            "A_kimjo_corrected": round(auc(sub["A_kimjo"].to_numpy(), y), 3),
        },
    }
    with open("results/burstiness_corrected.json", "w") as f:
        json.dump(out, f, indent=2)
    print("\n=== FINITE-SIZE-CORRECTED BURSTINESS (Kim-Jo A_n) ===")
    print(f"agents={out['n_agents']}  clean2class auto={out['clean2class']['n_auto']} human={out['clean2class']['n_human']}")
    print("confound (spearman vs n_gaps): raw CoV={raw_CoV}  B_raw={B_raw}  A_kimjo={A_kimjo_corrected}".format(**out["confound_check_spearman_vs_ngaps"]))
    print("mean by class:", json.dumps(out["mean_by_class"]))
    print("AUC predicts autonomous: raw CoV={raw_CoV}  B_raw={B_raw}  A_kimjo={A_kimjo_corrected}".format(**out["AUC_predicts_autonomous"]))


if __name__ == "__main__":
    main()
