from __future__ import annotations

import argparse
import json
import math
import os
import sys

import numpy as np
from scipy import stats
from statsmodels.stats.power import TTestIndPower

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_rigor as R

SEED = 42
EXACT_PERM_CAP = 1_000_000
MC_RESAMPLES = 99_999


def _mean_diff(x, y, axis=-1):
    return np.mean(x, axis=axis) - np.mean(y, axis=axis)


def _cohens_d(a, b):
    n1, n2 = len(a), len(b)
    sp = np.sqrt(((n1 - 1) * a.var(ddof=1) + (n2 - 1) * b.var(ddof=1)) / (n1 + n2 - 2))
    return (a.mean() - b.mean()) / sp if sp > 0 else 0.0


def exact_permutation(a, b, alpha=0.05):
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    n1, n2 = len(a), len(b)
    distinct = math.comb(n1 + n2, n1)
    feasible = distinct <= EXACT_PERM_CAP
    n_resamples = np.inf if feasible else MC_RESAMPLES
    res = stats.permutation_test(
        (a, b), _mean_diff,
        permutation_type="independent",
        vectorized=True,
        n_resamples=n_resamples,
        alternative="two-sided",
        random_state=SEED,
    )
    return {
        "statistic": float(res.statistic),
        "pvalue": float(res.pvalue),
        "mode": "exact" if feasible else "monte_carlo",
        "n_resamples_used": int(res.null_distribution.size),
        "distinct_permutations": float(distinct),
        "significant": bool(res.pvalue < alpha),
    }


def exact_rank_test(a, b, alpha=0.05):
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    n1, n2 = len(a), len(b)
    ties = (np.unique(np.concatenate([a, b])).size < n1 + n2)
    res = stats.mannwhitneyu(a, b, alternative="two-sided", method="exact")
    u1 = float(res.statistic)
    rank_biserial = 2.0 * u1 / (n1 * n2) - 1.0
    return {
        "U1": u1,
        "pvalue": float(res.pvalue),
        "method": "exact",
        "ties_present": bool(ties),
        "rank_biserial": float(rank_biserial),
        "significant": bool(res.pvalue < alpha),
        "note": "exact method is not tie-corrected; report ties_present" if ties else None,
    }


def effect_and_bootstrap(a, b, confidence_level=0.95):
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    d = _cohens_d(a, b)
    bs = stats.bootstrap(
        (a, b), _mean_diff,
        vectorized=True,
        paired=False,
        confidence_level=confidence_level,
        method="BCa",
        n_resamples=9999,
        random_state=SEED,
    )
    return {
        "collab_mean": float(a.mean()),
        "single_mean": float(b.mean()),
        "mean_diff": float(a.mean() - b.mean()),
        "cohens_d": float(d),
        "mean_diff_ci95": [float(bs.confidence_interval.low), float(bs.confidence_interval.high)],
        "mean_diff_se": float(bs.standard_error),
    }


def power_statement(n1, n2, observed_d, alpha=0.05, power=0.80):
    pw = TTestIndPower()
    ratio = n2 / n1
    required = {}
    for d in (0.3, 0.5):
        required[f"n_per_group_at_d{d}"] = float(
            pw.solve_power(effect_size=d, nobs1=None, alpha=alpha, power=power,
                           ratio=1.0, alternative="two-sided")
        )
    d_floor = float(pw.solve_power(effect_size=None, nobs1=n1, alpha=alpha, power=power,
                                   ratio=ratio, alternative="two-sided"))
    achieved = float(pw.power(effect_size=abs(observed_d), nobs1=n1, alpha=alpha,
                              ratio=ratio, alternative="two-sided"))
    return {
        "n_collab": int(n1),
        "n_single_matched": int(n2),
        "required_n": required,
        "min_detectable_d_at_80pct_power": d_floor,
        "achieved_power_at_observed_d": achieved,
    }


def robustness_panel(a, b, alpha=0.05):
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    perm = exact_permutation(a, b, alpha)
    rank = exact_rank_test(a, b, alpha)
    eff = effect_and_bootstrap(a, b)
    pwr = power_statement(len(a), len(b), eff["cohens_d"], alpha)
    underpowered = pwr["min_detectable_d_at_80pct_power"] > abs(eff["cohens_d"])
    return {
        "n_collab": int(len(a)),
        "n_single_matched": int(len(b)),
        "permutation_mean_diff": perm,
        "rank_test": rank,
        "effect_size": eff,
        "power": pwr,
        "underpowered": bool(underpowered),
        "headline_eligible": False,
        "framing": (
            "Supporting robustness panel beside the continuous model. Inference is exact "
            f"(permutation p={perm['pvalue']:.4f} via {perm['mode']}; rank-biserial "
            f"r={rank['rank_biserial']:.3f}, exact MWU p={rank['pvalue']:.4f}). "
            f"At n=({len(a)},{len(b)}) the design can resolve only d>="
            f"{pwr['min_detectable_d_at_80pct_power']:.2f} at 80% power, so a non-rejection "
            "is inconclusive rather than evidence of equivalence."
            if underpowered else
            "Supporting robustness panel beside the continuous model; exact inference "
            "corroborates the continuous estimate without driving the headline."
        ),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--start", default="2026-01-28")
    ap.add_argument("--end", default="2026-02-20")
    ap.add_argument("--exclude-dates", default=None)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--out", default="results/robustness_panel.json")
    args = ap.parse_args()

    excl = [d.strip() for d in args.exclude_dates.split(",")] if args.exclude_dates else None
    posts, comments = R.prepare(args.data_dir, args.start, args.end, None, excl)
    arrs = R.collab_arrays(posts, comments)

    out = {"window": [args.start, args.end], "alpha": args.alpha,
           "collab_meta": arrs["meta"], "panel": {}}
    for metric in ("biased", "neutral"):
        if metric in arrs:
            a, b = arrs[metric]
            out["panel"][metric] = R._safe(lambda a=a, b=b: robustness_panel(a, b, args.alpha))
    out["note"] = "'biased'=paper's single-agent-favoring metric; 'neutral'=collaboration-neutral"

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(json.dumps(out, indent=2, default=str))


if __name__ == "__main__":
    main()
