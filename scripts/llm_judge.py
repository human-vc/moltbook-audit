from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import polars as pl

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

SEED = 42
try:
    import run_rigor as R
    SEED = R.SEED
    log = R.log
except Exception:
    import logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
    log = logging.getLogger("judge")
    R = None

DIMS = ["correctness", "completeness", "justification", "actionability", "clarity", "overall"]

SCHEMA = {
    "type": "object",
    "properties": {**{d: {"type": "integer", "minimum": 1, "maximum": 5} for d in DIMS},
                   "rationale": {"type": "string"}},
    "required": DIMS + ["rationale"],
    "additionalProperties": False,
}

SYSTEM = (
    "You are an expert technical reviewer scoring the quality of the solution reached in a "
    "developer discussion thread. Judge only technical merit, not length or style: a short correct "
    "answer outscores a long vague one. Do not reward verbosity. Return only the requested JSON."
)

RUBRIC = (
    "Score the thread's solution on five dimensions, each an integer 1-5, then an overall 1-5.\n"
    "Steps: (1) identify the actual problem; (2) judge whether the solution would correctly resolve it; "
    "(3) check coverage of edge cases/caveats; (4) check whether claims are justified and steps executable; "
    "(5) assign each dimension by its anchors.\n"
    "correctness: 1=wrong/misleading ... 5=fully correct.\n"
    "completeness: 1=ignores most of the problem ... 5=covers all parts and edge cases.\n"
    "justification: 1=asserted without support ... 5=claims well supported.\n"
    "actionability: 1=not executable ... 5=concrete, runnable, correct.\n"
    "clarity: 1=incoherent ... 5=well organized and unambiguous.\n"
    "overall: holistic technical quality, 1-5.\n"
    'Return JSON: {"correctness":int,"completeness":int,"justification":int,'
    '"actionability":int,"clarity":int,"overall":int,"rationale":str}'
)


def _stratified(df, key, n, seed):
    if df.height <= n:
        return df
    rng = np.random.RandomState(seed)
    labels = df[key].to_numpy()
    total = len(labels)
    picks = []
    for lab in np.unique(labels):
        idx = np.where(labels == lab)[0]
        take = min(len(idx), max(1, int(round(n * len(idx) / total))))
        picks.append(rng.choice(idx, size=take, replace=False))
    sel = np.sort(np.concatenate(picks))
    return df[sel.tolist()]


def build(data_dir, start, end, sample, max_chars, out):
    posts, comments = R.prepare(data_dir, start, end, None, None)
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
                       pl.col("bodies").list.join("\n")], separator="\n").alias("thread_text"))
    th = th.with_columns(
        pl.col("thread_text").str.to_lowercase().str.contains("|".join(R.TECHNICAL_KEYWORDS)).alias("is_technical"),
        pl.col("thread_text").str.contains(r"```|`[^`]+`").alias("has_code"))
    th = th.filter(pl.col("is_technical")).with_columns(pl.col("thread_text").str.slice(0, max_chars))
    th = th.with_columns(
        pl.when(pl.col("n_contributors") <= 1).then(pl.lit("1"))
          .when(pl.col("n_contributors") == 2).then(pl.lit("2"))
          .when(pl.col("n_contributors") <= 4).then(pl.lit("3-4"))
          .when(pl.col("n_contributors") <= 9).then(pl.lit("5-9"))
          .otherwise(pl.lit("10+")).alias("contrib_bucket"))
    sub = _stratified(th, "contrib_bucket", sample, SEED)
    cols = ["post_id", "submolt", "n_contributors", "n_comments", "duration_min",
            "has_code", "contrib_bucket", "thread_text"]
    sub.select(cols).write_parquet(out)
    log.info("wrote %s threads to %s (buckets: %s)", sub.height, out,
               dict(zip(*[list(x) for x in np.unique(sub["contrib_bucket"].to_numpy(), return_counts=True)])))


def _parse(text):
    try:
        d = json.loads(text)
        return {k: int(d[k]) for k in DIMS if k in d}
    except Exception:
        return None


def _aggregate(samples):
    if not samples:
        return {**{d: None for d in DIMS}, "n_valid": 0, "overall_sd": None}
    out = {"n_valid": len(samples)}
    for d in DIMS:
        vals = [s[d] for s in samples if d in s]
        out[d] = float(np.mean(vals)) if vals else None
    ov = [s["overall"] for s in samples if "overall" in s]
    out["overall_sd"] = float(np.std(ov)) if len(ov) > 1 else 0.0
    return out


def judge(threads_path, model, k, max_model_len, temperature, out):
    from vllm import LLM, SamplingParams
    df = pl.read_parquet(threads_path)
    try:
        from vllm.sampling_params import GuidedDecodingParams
        sp = SamplingParams(n=k, temperature=temperature, top_p=0.95, max_tokens=512, seed=SEED,
                            guided_decoding=GuidedDecodingParams(json=SCHEMA))
    except Exception as e:
        log.warning("guided decoding unavailable (%s); relying on prompt + defensive parse", e)
        sp = SamplingParams(n=k, temperature=temperature, top_p=0.95, max_tokens=512, seed=SEED)
    llm = LLM(model=model, dtype="auto", max_model_len=max_model_len,
              gpu_memory_utilization=0.92, enable_prefix_caching=True)
    convos = [[{"role": "system", "content": SYSTEM},
               {"role": "user", "content": RUBRIC + "\n\n[THREAD]\n" + t}]
              for t in df["thread_text"].to_list()]
    outs = llm.chat(convos, sp)
    rows = []
    for pid, o in zip(df["post_id"].to_list(), outs):
        samples = [d for d in (_parse(c.text) for c in o.outputs) if d]
        rows.append({"post_id": pid, **_aggregate(samples)})
    scores = pl.DataFrame(rows)
    res = df.select(["post_id", "submolt", "n_contributors", "n_comments", "duration_min", "has_code"]).join(
        scores, on="post_id", how="left")
    res.write_parquet(out)
    log.info("judged %d threads with %s -> %s; mean valid samples=%.2f",
               res.height, model, out, float(np.nanmean(res["n_valid"].to_numpy())))


def reliability(scores_path, human_csv, out):
    from sklearn.metrics import cohen_kappa_score
    from scipy.stats import spearmanr
    sc = pl.read_parquet(scores_path)
    hu = pl.read_csv(human_csv)
    m = sc.join(hu, on="post_id", how="inner")
    report = {"n_compared": m.height}
    for d in DIMS:
        hcol = f"{d}_human"
        if hcol not in m.columns or d not in m.columns:
            continue
        a = np.round(m[d].to_numpy()).astype(int)
        b = np.round(m[hcol].to_numpy()).astype(int)
        rho, p = spearmanr(a, b)
        report[d] = {"weighted_kappa": float(cohen_kappa_score(a, b, weights="quadratic")),
                     "spearman_rho": float(rho), "spearman_p": float(p)}
    try:
        import krippendorff
        data = np.vstack([np.round(m["overall"].to_numpy()), np.round(m["overall_human"].to_numpy())])
        report["overall_krippendorff_alpha"] = float(
            krippendorff.alpha(reliability_data=data, level_of_measurement="ordinal"))
    except Exception as e:
        report["krippendorff_note"] = f"skipped: {e}"
    with open(out, "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=["build", "judge", "reliability"])
    ap.add_argument("--data-dir")
    ap.add_argument("--start", default="2026-02-02")
    ap.add_argument("--end", default="2026-05-28")
    ap.add_argument("--sample", type=int, default=2500)
    ap.add_argument("--max-chars", type=int, default=6000)
    ap.add_argument("--threads", default="results/judge_threads.parquet")
    ap.add_argument("--model", default="Qwen/Qwen3-32B")
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--max-model-len", type=int, default=8192)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--scores", default="results/llm_judge_scores.parquet")
    ap.add_argument("--human-csv")
    ap.add_argument("--out")
    args = ap.parse_args()

    if args.mode == "build":
        build(args.data_dir, args.start, args.end, args.sample, args.max_chars,
              args.out or args.threads)
    elif args.mode == "judge":
        judge(args.threads, args.model, args.k, args.max_model_len, args.temperature,
              args.out or "results/llm_judge_scores.parquet")
    else:
        reliability(args.scores, args.human_csv, args.out or "results/llm_judge_reliability.json")


if __name__ == "__main__":
    main()
