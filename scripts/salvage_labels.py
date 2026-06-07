from __future__ import annotations

import glob
import json
import sys

import numpy as np
import polars as pl
from scipy.stats import rankdata

sys.path.insert(0, "scripts")
import run_rigor as R

DD = "moltbook-observatory-archive/moltbook-observatory-archive/data"

# high-precision first-person AI self-disclosure (positive autonomy label)
PATTERNS = [
    "as an ai", "i am an ai", "i'm an ai", "as a language model", "as an llm",
    "large language model", "ai language model", "i'm an ai assistant",
    "i cannot assist with", "i can't help with that", "i cannot fulfill",
    "i am not able to provide", "i'm unable to provide", "as an ai,",
]
REGEX = "(?i)" + "|".join(p.replace(" ", r"\s+") for p in PATTERNS)


def auc(score, label):
    s = np.asarray(score, float); y = np.asarray(label, bool)
    n = len(y); n1 = int(y.sum()); n0 = n - n1
    if n1 == 0 or n0 == 0:
        return None
    r = rankdata(s)
    return float((r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def main():
    posts, comments = R.prepare(DD, "2026-01-28", "2026-02-20", None, None)

    ptext = posts.select(["agent_id",
                          (pl.col("title").fill_null("") + " " + pl.col("content").fill_null("")).alias("t")])
    ctext = comments.select(["agent_id", pl.col("content").fill_null("").alias("t")])
    alltext = pl.concat([ptext, ctext])
    matched = alltext.filter(pl.col("t").str.contains(REGEX))
    disc_agents = set(matched["agent_id"].unique().to_list())
    R.log.info("self-disclosure: %d matching messages, %d distinct agents", matched.height, len(disc_agents))
    print("\n--- 6 sample self-disclosure snippets (precision eyeball) ---")
    for t in matched["t"].unique().to_list()[:6]:
        print("  •", t[:160].replace("\n", " "))

    cov = pl.from_pandas(R.autonomy_cov(posts, comments)[["agent_id", "cov", "n_events"]])
    ag = pl.scan_parquet(sorted(glob.glob(DD + "/agents/*.parquet"))).select(
        ["id", "is_claimed", "owner_x_handle"]).collect().with_columns(pl.col("id").cast(pl.Utf8).alias("agent_id"))

    d = cov.join(ag, on="agent_id", how="left").with_columns([
        pl.col("agent_id").is_in(list(disc_agents)).alias("self_ai"),
        (pl.col("owner_x_handle").is_not_null() & (pl.col("owner_x_handle").cast(pl.Utf8).str.len_chars() > 0)).alias("has_x"),
    ])
    dd = d.to_pandas()

    self_ai = dd["self_ai"].to_numpy()
    has_x = dd["has_x"].to_numpy()
    claimed = dd["is_claimed"].fill_null(False).to_numpy().astype(bool) if hasattr(dd["is_claimed"], "fill_null") else dd["is_claimed"].fillna(False).to_numpy().astype(bool)
    cov_arr = dd["cov"].to_numpy()

    out = {
        "n_scored_agents": int(len(dd)),
        "n_self_disclosed_AI_scored": int(self_ai.sum()),
        "n_x_linked_scored": int(has_x.sum()),
        "mean_cov_self_AI": float(cov_arr[self_ai].mean()) if self_ai.sum() else None,
        "mean_cov_not_self_AI": float(cov_arr[~self_ai].mean()),
        "mean_cov_x_linked_human": float(cov_arr[has_x].mean()),
        # filter says LOW cov = autonomous; so for autonomous label, score = -cov should give AUC>0.5 if valid
        "AUC_negcov_vs_selfAI_full": auc(-cov_arr, self_ai),
    }

    # clean two-class test: self-disclosed AI (autonomous) vs X-linked & not-self-AI (human)
    mask = self_ai | (has_x & ~self_ai)
    y_auto = self_ai[mask]
    out["clean2class_n_auto"] = int(y_auto.sum())
    out["clean2class_n_human"] = int((~y_auto).sum())
    out["AUC_negcov_selfAI_vs_xhuman"] = auc(-cov_arr[mask], y_auto)
    out["interpretation"] = (
        "AUC>0.6 => CoV has real validity against clean labels (salvageable); "
        "AUC~0.5 => no signal; AUC<0.5 => inverted (CoV invalid as autonomy proxy)")

    with open("results/salvage_labels.json", "w") as f:
        json.dump(out, f, indent=2)
    print("\n=== SALVAGE STEP 1: CoV vs clean labels ===")
    for k, v in out.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
