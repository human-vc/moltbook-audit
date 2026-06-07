from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
from scipy import stats, integrate
from statsmodels.stats.weightstats import ttost_ind
from statsmodels.stats.power import TTestIndPower

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_rigor as R


def bf10_jzs(t, n1, n2=None, r=0.707):
    df = (n1 - 1) if n2 is None else (n1 + n2 - 2)
    nstar = n1 if n2 is None else (n1 * n2) / (n1 + n2)

    def integrand(g):
        return ((1 + nstar * g * r ** 2) ** -0.5
                * (1 + t ** 2 / (df * (1 + nstar * g * r ** 2))) ** (-(df + 1) / 2)
                * (2 * np.pi) ** -0.5 * g ** -1.5 * np.exp(-1 / (2 * g)))

    num, _ = integrate.quad(integrand, 0, np.inf)
    return num / (1 + t ** 2 / df) ** (-(df + 1) / 2)


def audit_pair(a, b, sesoi_d=0.5, alpha=0.05, power=0.80):
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    n1, n2 = len(a), len(b)
    sp = np.sqrt(((n1 - 1) * a.var(ddof=1) + (n2 - 1) * b.var(ddof=1)) / (n1 + n2 - 2))
    d = (a.mean() - b.mean()) / sp if sp > 0 else 0.0
    bound = sesoi_d * sp

    p_tost, _lower, _upper = ttost_ind(a, b, -bound, bound, usevar="unequal")
    t = stats.ttest_ind(a, b, equal_var=False).statistic
    bf10 = bf10_jzs(t, n1, n2)

    pw = TTestIndPower()
    n_small, n_large = min(n1, n2), max(n1, n2)
    ratio = n_large / n_small
    d_floor = pw.solve_power(effect_size=None, nobs1=n_small, alpha=alpha,
                             power=power, ratio=ratio, alternative="two-sided")
    achieved = pw.power(effect_size=abs(d), nobs1=n_small, alpha=alpha,
                        ratio=ratio, alternative="two-sided")

    equiv = bool(p_tost < alpha)
    if equiv:
        verdict = "equivalent: no meaningful effect (TOST rejects presence)"
    elif bf10 > 3:
        verdict = "difference supported (BF10>3)"
    else:
        verdict = f"inconclusive/underpowered: can detect only d>={d_floor:.2f}"

    return {"n_collab": n1, "n_single": n2,
            "cohens_d": round(float(d), 3),
            "sesoi_d": sesoi_d, "sesoi_raw": round(float(bound), 4),
            "tost_p": round(float(p_tost), 4), "equivalence_declared": equiv,
            "bf10": round(float(bf10), 3), "bf01": round(float(1 / bf10), 3),
            "detectable_d_at_80pct_power": round(float(d_floor), 3),
            "achieved_power_at_observed_d": round(float(achieved), 3),
            "verdict": verdict}


def hr_rope(posts, rope=(0.9, 1.1111), min_adopters=8, max_cascades=800):
    from lifelines import CoxPHFitter
    fa = R._cascade_adoptions(posts, min_adopters, max_cascades)
    df = R._gap_table(fa)
    cph = CoxPHFitter(penalizer=0.01)
    cph.fit(df, duration_col="gap", event_col="event", cluster_col="cascade", show_progress=False)
    s = cph.summary.loc["exposure"]
    hr = float(np.exp(cph.params_["exposure"]))
    lo, hi = float(s["exp(coef) lower 95%"]), float(s["exp(coef) upper 95%"])
    inside = rope[0] <= lo and hi <= rope[1]
    return {"hazard_ratio": round(hr, 4), "hr_ci95": [round(lo, 4), round(hi, 4)],
            "rope": list(rope), "n_cascades": int(df["cascade"].nunique()),
            "n_adoptions": int(len(df)),
            "verdict": ("no reinforcement: equivalent to HR=1 (CI within ROPE)" if inside
                        else "inconclusive: HR CI exceeds the ROPE")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--start", default="2026-01-28")
    ap.add_argument("--end", default="2026-02-20")
    ap.add_argument("--exclude-dates", default=None)
    ap.add_argument("--sesoi-d", type=float, default=0.5)
    ap.add_argument("--out", default="results/equivalence.json")
    args = ap.parse_args()

    excl = [d.strip() for d in args.exclude_dates.split(",")] if args.exclude_dates else None
    posts, comments = R.prepare(args.data_dir, args.start, args.end, None, excl)

    arrs = R.collab_arrays(posts, comments)
    out = {"window": [args.start, args.end], "sesoi_d": args.sesoi_d,
           "collab_meta": arrs["meta"], "collaboration": {}}
    if "biased" in arrs:
        for metric in ("biased", "neutral"):
            a, b = arrs[metric]
            out["collaboration"][metric] = audit_pair(a, b, sesoi_d=args.sesoi_d)
    out["diffusion_hr"] = R._safe(lambda: hr_rope(posts))

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(json.dumps(out, indent=2, default=str))


if __name__ == "__main__":
    main()
