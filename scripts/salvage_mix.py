from __future__ import annotations

import glob
import json
import sys

import numpy as np
import polars as pl
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

sys.path.insert(0, "scripts")
import run_rigor as R
from salvage_classifier import features, REGEX, DD

COLS = ["cov", "log_n", "log_gap_h", "hour_entropy", "active_hours", "round_tick_frac", "gap_skew"]


def shares(prob, nev, thr=0.5):
    auto = prob >= thr
    return {"account_share_auto": round(float(auto.mean()), 3),
            "activity_share_auto": round(float(nev[auto].sum() / nev.sum()), 3)}


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
    Xall = np.nan_to_num(d.select(COLS).to_numpy().astype(float), nan=0.0, posinf=0.0, neginf=0.0)
    nev = np.expm1(d["log_n"].to_numpy())
    self_ai = d["self_ai"].to_numpy()
    has_x = d["has_x"].to_numpy()

    train = self_ai | (has_x & ~self_ai)
    Xtr, ytr = Xall[train], self_ai[train].astype(int)

    gb = GradientBoostingClassifier(random_state=42)
    lr = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced"))
    cv = StratifiedKFold(5, shuffle=True, random_state=42)

    def pop_prob(est):
        p = np.full(len(Xall), np.nan)
        oof = cross_val_predict(est, Xtr, ytr, cv=cv, method="predict_proba")[:, 1]
        p[train] = oof
        est.fit(Xtr, ytr)
        rest = ~train
        p[rest] = est.predict_proba(Xall[rest])[:, 1]
        return p

    p_gb = pop_prob(GradientBoostingClassifier(random_state=42))
    p_lr = pop_prob(make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced")))

    out = {
        "n_scored_agents": int(len(Xall)),
        "training_prior_auto": round(float(ytr.mean()), 3),
        "n_labeled_train": int(train.sum()),
        "CAVEATS": ["arbitrary label prior (label availability, not truth) => absolute % unreliable",
                    "X-linked 'human' may be human-OWNED scheduled bots, not real-time steering",
                    "self-disclosed AIs are a subset; extrapolation to unlabeled ~middle is uncertain",
                    "report as a RANGE bracketed by conservative(GBM) and prior-neutral(balanced-LR), not a point estimate"],
        "conservative_gbm": {t: shares(p_gb, nev, t) for t in (0.3, 0.5, 0.7)},
        "prior_neutral_lr": {t: shares(p_lr, nev, t) for t in (0.3, 0.5, 0.7)},
        "original_filter_for_contrast": "single-CoV>0.75 labeled 72.7% of agents 'human-steered' (inverted polarity, invalid)",
    }
    with open("results/salvage_mix.json", "w") as f:
        json.dump(out, f, indent=2)

    print("\n=== CORRECTED MIX ESTIMATE (range, NOT a point estimate) ===")
    print(f"scored agents={out['n_scored_agents']}  training prior auto={out['training_prior_auto']}")
    for name, p in [("GBM (conservative, training prior)", p_gb), ("balanced-LR (prior-neutral)", p_lr)]:
        s5 = shares(p, nev, 0.5)
        print(f"  {name}: @0.5  accounts auto={s5['account_share_auto']}  ACTIVITY auto={s5['activity_share_auto']}")
    print("  full threshold sweep in results/salvage_mix.json")
    print("  contrast: original single-CoV filter called 72.7% 'human-steered' (inverted/invalid)")


if __name__ == "__main__":
    main()
