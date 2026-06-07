from __future__ import annotations

import json

import numpy as np
import pandas as pd
import patsy
import polars as pl
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.outliers_influence import variance_inflation_factor

SEED = 42
CTRL = "log_comments + duration_min + has_code + log_charlen"
FORM = "n_contributors + " + CTRL


def load():
    js = pl.read_parquet("results/llm_judge_scores.parquet").select(
        ["post_id", "submolt", "n_contributors", "n_comments", "duration_min", "has_code", "overall"]
    ).rename({"overall": "overall_judge"})
    th = pl.read_parquet("results/judge_threads.parquet").select(["post_id", "thread_text"]).with_columns(
        pl.col("thread_text").str.len_chars().alias("charlen")).drop("thread_text")
    hu = pl.read_csv("results/human_labels.csv").select(["post_id", "overall_human"])
    base = js.join(th, on="post_id", how="inner")
    judge = base.drop_nulls(["overall_judge", "n_contributors"]).to_pandas()
    human = base.join(hu, on="post_id", how="inner").drop_nulls(["overall_human", "n_contributors"]).to_pandas()
    for d in (judge, human):
        d["log_comments"] = np.log1p(d["n_comments"])
        d["log_charlen"] = np.log1p(d["charlen"])
        d["has_code"] = d["has_code"].astype(int)
    return human, judge


def fit(frame, dv, rhs=FORM):
    return smf.ols(f"{dv} ~ {rhs}", data=frame).fit(
        cov_type="cluster", cov_kwds={"groups": frame["submolt"]})


def row(frame, dv, rhs=FORM, key="n_contributors"):
    m = fit(frame, dv, rhs)
    ci = m.conf_int().loc[key]
    return dict(slope=float(m.params[key]), se=float(m.bse[key]), p=float(m.pvalues[key]),
                lo=float(ci[0]), hi=float(ci[1]), n=int(m.nobs),
                G=int(frame["submolt"].nunique()))


def influence(frame, dv, focal="n_contributors"):
    m = smf.ols(f"{dv} ~ {FORM}", data=frame).fit()
    infl = m.get_influence()
    sf = infl.summary_frame()
    n = int(m.nobs)
    cooks = infl.cooks_distance[0]
    dfb = sf[f"dfb_{focal}"].to_numpy()
    flag = sorted(set(np.where(cooks > 4.0 / n)[0]) | set(np.where(np.abs(dfb) > 2.0 / np.sqrt(n))[0]))
    dropped = row(frame.drop(frame.index[flag]), dv) if flag else None
    return {"n_flagged": len(flag), "frac_flagged": round(len(flag) / n, 4), "refit_dropping_flagged": dropped}


def ladder(frame, dv):
    out = {"full": row(frame, dv)}
    for cap in (50, 20):
        out[f"drop_contrib_gt_{cap}"] = row(frame[frame["n_contributors"] <= cap], dv)
    for q in (0.99, 0.95):
        thr = frame["n_contributors"].quantile(q)
        out[f"keep_contrib_below_p{int(q*100)}"] = row(frame[frame["n_contributors"] <= thr], dv)
    cthr = frame["charlen"].quantile(0.90)
    out["drop_top_decile_charlen"] = row(frame[frame["charlen"] <= cthr], dv)
    return out


def functional_form(frame, dv):
    m_lin = fit(frame, dv, "n_contributors + " + CTRL)
    m_log = fit(frame, dv, "np.log1p(n_contributors) + " + CTRL)
    m_quad = fit(frame, dv, "n_contributors + I(n_contributors ** 2) + " + CTRL)
    w = m_quad.wald_test("I(n_contributors ** 2) = 0", use_f=False)
    return {
        "linear_slope": float(m_lin.params["n_contributors"]), "linear_aic": float(m_lin.aic),
        "log_slope": float(m_log.params["np.log1p(n_contributors)"]),
        "log_p": float(m_log.pvalues["np.log1p(n_contributors)"]), "log_aic": float(m_log.aic),
        "quad_coef": float(m_quad.params["I(n_contributors ** 2)"]),
        "quad_robust_p": float(np.squeeze(w.pvalue)),
        "note": "log AIC lower => diminishing-returns fit preferred; negative quad coef => concavity/saturation",
    }


def leave_one_cluster_out(frame, dv, key="n_contributors"):
    slopes = []
    for g in frame["submolt"].unique():
        sub = frame[frame["submolt"] != g]
        if sub["submolt"].nunique() < 2:
            continue
        try:
            slopes.append(float(fit(sub, dv).params[key]))
        except Exception:
            continue
    s = np.array(slopes)
    return {"min": float(s.min()), "max": float(s.max()), "all_positive": bool((s > 0).all()), "n_clusters_dropped": len(s)}


def cluster_bootstrap(frame, dv, key="n_contributors", B=2000):
    rng = np.random.default_rng(SEED)
    groups = frame["submolt"].unique()
    by = {g: frame[frame["submolt"] == g] for g in groups}
    est = []
    for _ in range(B):
        samp = pd.concat([by[g] for g in rng.choice(groups, size=len(groups), replace=True)], ignore_index=True)
        try:
            est.append(fit(samp, dv).params[key])
        except Exception:
            continue
    est = np.asarray(est)
    return {"boot_lo": float(np.percentile(est, 2.5)), "boot_hi": float(np.percentile(est, 97.5)),
            "boot_p": float(2 * min((est <= 0).mean(), (est >= 0).mean())),
            "n_clusters": int(len(groups)), "B_ok": int(len(est))}


def vif_table(frame):
    X = patsy.dmatrix(FORM, frame, return_type="dataframe")
    return {c: round(float(variance_inflation_factor(X.values, i)), 2)
            for i, c in enumerate(X.columns) if c != "Intercept"}


def fractional_logit(frame, dv):
    f = frame.copy()
    f["dv01"] = (f[dv] - 1.0) / 4.0
    m = smf.glm("dv01 ~ " + FORM, data=f, family=sm.families.Binomial()).fit(
        cov_type="cluster", cov_kwds={"groups": f["submolt"]})
    return {"coef": float(m.params["n_contributors"]), "p": float(m.pvalues["n_contributors"])}


def analyze(frame, dv):
    return {
        "n": int(len(frame)), "G": int(frame["submolt"].nunique()),
        "contrib_max": int(frame["n_contributors"].max()),
        "ladder": ladder(frame, dv),
        "influence": influence(frame, dv),
        "functional_form": functional_form(frame, dv),
        "leave_one_cluster_out": leave_one_cluster_out(frame, dv),
        "cluster_bootstrap": cluster_bootstrap(frame, dv),
        "vif": vif_table(frame),
        "fractional_logit": fractional_logit(frame, dv),
        "bivariate_slope": float(smf.ols(f"{dv} ~ n_contributors", data=frame).fit(
            cov_type="cluster", cov_kwds={"groups": frame["submolt"]}).params["n_contributors"]),
    }


def main():
    human, judge = load()
    out = {"human_n303": analyze(human, "overall_human"),
           "judge_full": analyze(judge, "overall_judge")}
    with open("results/collab_robustness.json", "w") as f:
        json.dump(out, f, indent=2)
    for k, r in out.items():
        ld = r["ladder"]
        print(f"\n=== {k} (n={r['n']}, G={r['G']}, max_contrib={r['contrib_max']}) ===")
        print(f"  full slope      {ld['full']['slope']:+.5f}  p={ld['full']['p']:.2e}")
        print(f"  drop>50         {ld['drop_contrib_gt_50']['slope']:+.5f}  p={ld['drop_contrib_gt_50']['p']:.2e}")
        print(f"  drop>20         {ld['drop_contrib_gt_20']['slope']:+.5f}  p={ld['drop_contrib_gt_20']['p']:.2e}")
        print(f"  drop top-decile len {ld['drop_top_decile_charlen']['slope']:+.5f}  p={ld['drop_top_decile_charlen']['p']:.2e}")
        print(f"  drop influential    {r['influence']['refit_dropping_flagged']}")
        ff = r["functional_form"]
        print(f"  log slope {ff['log_slope']:+.4f} p={ff['log_p']:.2e} | AIC lin {ff['linear_aic']:.1f} vs log {ff['log_aic']:.1f} | quad coef {ff['quad_coef']:+.2e} p={ff['quad_robust_p']:.3f}")
        cb = r["cluster_bootstrap"]
        print(f"  cluster bootstrap CI [{cb['boot_lo']:+.5f}, {cb['boot_hi']:+.5f}] p={cb['boot_p']:.3f}  (G={cb['n_clusters']})")
        loco = r["leave_one_cluster_out"]
        print(f"  leave-1-cluster-out slope range [{loco['min']:+.5f}, {loco['max']:+.5f}] all_positive={loco['all_positive']}")
        print(f"  VIF {r['vif']}")
        print(f"  fractional-logit coef {r['fractional_logit']['coef']:+.4f} p={r['fractional_logit']['p']:.2e}")


if __name__ == "__main__":
    main()
