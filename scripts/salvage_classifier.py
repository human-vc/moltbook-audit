from __future__ import annotations

import glob
import json
import sys

import numpy as np
import polars as pl
from scipy.stats import entropy as shannon, skew
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

sys.path.insert(0, "scripts")
import run_rigor as R

DD = "moltbook-observatory-archive/moltbook-observatory-archive/data"
PATTERNS = ["as an ai", "i am an ai", "i'm an ai", "as a language model", "as an llm",
            "large language model", "ai language model", "i'm an ai assistant",
            "i cannot assist with", "i can't help with that", "i cannot fulfill",
            "i am not able to provide", "i'm unable to provide", "as an ai,"]
REGEX = "(?i)" + "|".join(p.replace(" ", r"\s+") for p in PATTERNS)
TICKS = np.array([60, 300, 600, 900, 1800, 3600, 7200, 86400], float)


def features(posts, comments):
    ev = pl.concat([posts.select(["agent_id", "created_at"]),
                    comments.select(["agent_id", "created_at"])]).drop_nulls().sort(["agent_id", "created_at"])
    g = (ev.with_columns(pl.col("created_at").dt.hour().alias("h"))
         .group_by("agent_id")
         .agg(n=pl.len(),
              gaps=pl.col("created_at").diff().dt.total_seconds().drop_nulls(),
              hours=pl.col("h"))
         .filter(pl.col("n") >= 5))
    rows = []
    for aid, n, gaps, hours in zip(g["agent_id"], g["n"], g["gaps"], g["hours"]):
        a = np.array([x for x in gaps if x and x > 0], float)
        if len(a) < 4:
            continue
        m = a.mean()
        hh = np.bincount(np.array(hours, int), minlength=24).astype(float)
        tick = np.min(np.abs(a[:, None] - TICKS[None, :]) / TICKS[None, :], axis=1)
        rows.append((str(aid), float(a.std() / m) if m else 0.0, float(np.log1p(n)),
                     float(np.log1p(m / 3600)),
                     float(shannon(hh + 1e-9) / np.log(24)),
                     int((hh > 0).sum()),
                     float((tick < 0.02).mean()),
                     float(skew(np.log1p(a))) if len(a) > 2 else 0.0))
    return pl.DataFrame(rows, schema=["agent_id", "cov", "log_n", "log_gap_h", "hour_entropy",
                                      "active_hours", "round_tick_frac", "gap_skew"], orient="row")


def auc_cv(X, y, est):
    cv = StratifiedKFold(5, shuffle=True, random_state=42)
    return cross_val_score(est, X, y, cv=cv, scoring="roc_auc")


def main():
    posts, comments = R.prepare(DD, "2026-01-28", "2026-02-20", None, None)
    ptext = posts.select(["agent_id", (pl.col("title").fill_null("") + " " + pl.col("content").fill_null("")).alias("t")])
    ctext = comments.select(["agent_id", pl.col("content").fill_null("").alias("t")])
    disc = set(pl.concat([ptext, ctext]).filter(pl.col("t").str.contains(REGEX))["agent_id"].unique().to_list())

    feat = features(posts, comments)
    ag = pl.scan_parquet(sorted(glob.glob(DD + "/agents/*.parquet"))).select(["id", "owner_x_handle"]).collect().with_columns(
        pl.col("id").cast(pl.Utf8).alias("agent_id"))
    d = feat.join(ag, on="agent_id", how="left").with_columns([
        pl.col("agent_id").is_in(list(disc)).alias("self_ai"),
        (pl.col("owner_x_handle").is_not_null() & (pl.col("owner_x_handle").cast(pl.Utf8).str.len_chars() > 0)).alias("has_x"),
    ])
    # clean two-class: self-disclosed AI (autonomous=1) vs X-linked & not-self-AI (human=0)
    sub = d.filter(pl.col("self_ai") | (pl.col("has_x") & ~pl.col("self_ai")))
    y = sub["self_ai"].to_numpy().astype(int)
    cols = ["cov", "log_n", "log_gap_h", "hour_entropy", "active_hours", "round_tick_frac", "gap_skew"]
    X = np.nan_to_num(sub.select(cols).to_numpy().astype(float), nan=0.0, posinf=0.0, neginf=0.0)

    out = {"n_auto": int(y.sum()), "n_human": int((~y.astype(bool)).sum()),
           "feature_means_auto": {c: round(float(sub.filter(pl.col("self_ai")).select(c).to_numpy().mean()), 3) for c in cols},
           "feature_means_human": {c: round(float(sub.filter(~pl.col("self_ai")).select(c).to_numpy().mean()), 3) for c in cols}}

    # single-feature AUCs (flipped where needed): use each feature alone
    from scipy.stats import rankdata
    def auc1(s, yy):
        s = np.asarray(s, float); n1 = yy.sum(); n0 = len(yy) - n1
        r = rankdata(s); return float((r[yy == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))
    out["single_feature_AUC"] = {c: round(auc1(X[:, i], y), 3) for i, c in enumerate(cols)}

    def cv_both(Xm, tag):
        lr = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced"))
        gb = GradientBoostingClassifier(random_state=42)
        out[f"cv_auc_logistic_{tag}"] = round(float(np.mean(auc_cv(Xm, y, lr))), 3)
        out[f"cv_auc_gbm_{tag}"] = round(float(np.mean(auc_cv(Xm, y, gb))), 3)

    cv_both(X, "all")
    no_n = [i for i, c in enumerate(cols) if c != "log_n"]
    cv_both(X[:, no_n], "no_logn")
    cad_circ = [i for i, c in enumerate(cols) if c in ("cov", "hour_entropy", "active_hours", "round_tick_frac")]
    cv_both(X[:, cad_circ], "cadence_circadian_only")

    with open("results/salvage_classifier.json", "w") as f:
        json.dump(out, f, indent=2)
    print("\n=== SALVAGE STEP 2: multi-feature timing classifier (CV) vs clean labels ===")
    print(f"n_auto(self-AI)={out['n_auto']}  n_human(X-linked)={out['n_human']}")
    print("single-feature AUC (1=autonomous):", out["single_feature_AUC"])
    print("auto feature means:  ", out["feature_means_auto"])
    print("human feature means: ", out["feature_means_human"])
    print(f"CV AUC all features:        logistic={out['cv_auc_logistic_all']}  gbm={out['cv_auc_gbm_all']}")
    print(f"CV AUC no event-count:      logistic={out['cv_auc_logistic_no_logn']}  gbm={out['cv_auc_gbm_no_logn']}")
    print(f"CV AUC cadence+circadian:   logistic={out['cv_auc_logistic_cadence_circadian_only']}  gbm={out['cv_auc_gbm_cadence_circadian_only']}")


if __name__ == "__main__":
    main()
