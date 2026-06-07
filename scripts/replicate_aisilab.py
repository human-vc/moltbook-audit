from __future__ import annotations

import json
import sys

import numpy as np
import polars as pl

sys.path.insert(0, "scripts")
import run_rigor as R

PARQUET = "data_aisilab/train.parquet"
COV_THRESHOLD = 0.75


def _naive(col):
    return (pl.col(col)
            .str.to_datetime(format="%Y-%m-%dT%H:%M:%S%.f%:z", time_unit="us", strict=False)
            .dt.replace_time_zone(None))


def load_frames():
    df = pl.read_parquet(PARQUET)
    posts = df.select([
        pl.col("post_id").alias("id"),
        pl.col("author_id").cast(pl.Utf8).alias("agent_id"),
        pl.col("submolt_id").alias("submolt"),
        pl.col("title"),
        pl.col("content"),
        _naive("created_at").alias("created_at"),
    ])

    rows = []
    pids = df["post_id"].to_list()
    cjson = df["comments"].to_list()

    def walk(lst, post_id, parent_id):
        for c in lst:
            cid = c.get("id")
            rows.append((cid, post_id, str(c.get("author_id")), parent_id,
                         c.get("content"), c.get("created_at")))
            reps = c.get("replies")
            if reps:
                walk(reps, post_id, cid)

    for pid, cj in zip(pids, cjson):
        if not cj:
            continue
        try:
            walk(json.loads(cj), pid, None)
        except Exception:
            continue

    comments = pl.DataFrame(
        rows, schema=["id", "post_id", "agent_id", "parent_id", "content", "created_at"],
        orient="row",
    ).with_columns(_naive("created_at").alias("created_at"))
    return posts, comments


def human_share(posts, comments):
    cov = R.autonomy_cov(posts, comments)
    if not len(cov):
        return {"result": "no scored agents"}
    human = set(cov.loc[cov["cov"] > COV_THRESHOLD, "agent_id"])
    auton = set(cov.loc[cov["cov"] <= COV_THRESHOLD, "agent_id"])
    cl = comments.filter(pl.col("agent_id").is_in(list(human | auton)))
    n_cl = cl.height
    n_hu = cl.filter(pl.col("agent_id").is_in(list(human))).height
    return {
        "cov_threshold": COV_THRESHOLD,
        "n_agents_scored": int(len(cov)),
        "n_human_steered": int(len(human)),
        "n_autonomous": int(len(auton)),
        "comments_classifiable": int(n_cl),
        "human_steered_comment_share": round(n_hu / n_cl, 4) if n_cl else None,
    }


def overlap_with_main():
    import glob
    fs = sorted(glob.glob("moltbook-observatory-archive/moltbook-observatory-archive/data/posts/*.parquet"))
    if not fs:
        return {"error": "main posts not found"}
    main_ids = set(pl.scan_parquet(fs).select("id").collect()["id"].to_list())
    ais_ids = set(pl.read_parquet(PARQUET, columns=["post_id"])["post_id"].to_list())
    inter = len(main_ids & ais_ids)
    frac = round(inter / len(ais_ids), 3)
    return {"main_ids": len(main_ids), "aisilab_ids": len(ais_ids), "post_id_overlap": inter,
            "overlap_frac_of_aisilab": frac,
            "interpretation": (f"NOT independent: {int(frac*100)}% of aisilab posts are the SAME posts as the "
                               "main archive; this is collector/schema robustness only, not independent replication")
            if frac > 0.5 else "largely disjoint: an independent sample"}


def main():
    posts, comments = load_frames()
    R.log.info("aisilab: posts=%d comments=%d agents=%d window=%s..%s",
               posts.height, comments.height, posts["agent_id"].n_unique(),
               str(posts["created_at"].min()), str(posts["created_at"].max()))

    edges = R.build_edges(posts, comments)
    g = R.build_graph(edges)

    out = {
        "source": "aisilab/moltbook-files (independent collector)",
        "counts": {"posts": posts.height, "comments": comments.height,
                   "agents": int(posts["agent_id"].n_unique()),
                   "graph_nodes": g.vcount(), "graph_edges": g.ecount()},
        "id_overlap_vs_main": overlap_with_main(),
        "task1": R.task1(g),
        "task1_nulls": R.task1_nulls(g, n_iter=50, n_jobs=4),
        "communities": R.task1_leiden(g, n_null=20),
        "cascades": R.task2(posts, n_shuffle=20, gof_iters=0),
        "human_share": human_share(posts, comments),
    }
    with open("results/replication_aisilab.json", "w") as f:
        json.dump(out, f, indent=2, default=str)

    rec = out["task1_nulls"].get("reciprocity", {})
    com = out["communities"]
    casc = out["cascades"].get("powerlaw", {})
    print("\n===== CROSS-COLLECTOR REPLICATION: aisilab/moltbook-files =====")
    print(f"posts={out['counts']['posts']}  comments={out['counts']['comments']}  "
          f"graph={g.vcount()}n/{g.ecount()}e  window 12d (Jan27-Feb7)")
    print(f"id overlap vs main: {out['id_overlap_vs_main']}")
    print(f"reciprocity obs={rec.get('observed'):.4f} null={rec.get('null_mean'):.4f} "
          f"ratio={rec.get('observed')/rec.get('null_mean'):.1f}x z={rec.get('z'):.1f}")
    mv = com.get("modularity_vs_null", {})
    print(f"modularity Q={com.get('modularity'):.3f} null={mv.get('null_mean'):.3f} z={mv.get('z')}  "
          f"n_comm={com.get('n_communities')}")
    print(f"cascades n={out['cascades'].get('n_cascades')}  verdict: {casc.get('verdict')}")
    print(f"human-steered comment share: {out['human_share'].get('human_steered_comment_share')}")


if __name__ == "__main__":
    main()
