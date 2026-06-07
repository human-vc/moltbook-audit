from __future__ import annotations

import argparse
import json
import os
import random
import sys

import numpy as np
import igraph as ig
import graph_tool.all as gt
from joblib import Parallel, delayed

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_rigor as R

SEED = R.SEED


def _igraph_to_gt(g_ig):
    g = gt.Graph(directed=True)
    g.add_vertex(g_ig.vcount())
    wp = g.new_ep("double")
    edges = [(e.source, e.target, w) for e, w in zip(g_ig.es, g_ig.es["weight"])]
    g.add_edge_list(edges, eprops=[wp])
    g.ep["weight"] = wp
    return g


def _leiden_Q(g_ig):
    p = g_ig.community_leiden(objective_function="modularity", weights="weight", n_iterations=3)
    return float(g_ig.modularity(p.membership, weights="weight", directed=True))


def _null_worker(b, bg_adj, out_degs, in_degs, seed):
    gt.seed_rng(seed)
    u = gt.generate_sbm(b, bg_adj, out_degs, in_degs, directed=True)
    edges = [(int(e.source()), int(e.target())) for e in u.edges()]
    gi = ig.Graph(n=u.num_vertices(), edges=edges, directed=True)
    gi.es["weight"] = 1
    return _leiden_Q(gi)


def dcsbm_null(g_ig, n_null=100, n_jobs=-1, weighted=True):
    g = _igraph_to_gt(g_ig)
    Q_obs = _leiden_Q(g_ig)

    sargs = dict(deg_corr=True)
    if weighted:
        sargs |= dict(recs=[g.ep.weight], rec_types=["discrete-geometric"])
    nested = gt.minimize_nested_blockmodel_dl(g, state_args=sargs)
    ndc = gt.minimize_nested_blockmodel_dl(g, state_args=dict(deg_corr=False))

    state = nested.get_levels()[0]
    b = np.array(state.b.a)
    bg_adj = gt.adjacency(state.get_bg(), state.get_ers()).T
    out_d = g.degree_property_map("out").a
    in_d = g.degree_property_map("in").a

    nulls = Parallel(n_jobs=n_jobs, prefer="processes")(
        delayed(_null_worker)(b, bg_adj, out_d, in_d, SEED + 7000 + i) for i in range(n_null))
    a = np.array([x for x in nulls if x is not None])

    out = {
        "modularity_observed": Q_obs,
        "n_blocks_inferred": int(state.get_nonempty_B()),
        "dl_deg_corr": float(nested.entropy()),
        "dl_non_deg_corr": float(ndc.entropy()),
        "deg_corr_preferred": bool(nested.entropy() < ndc.entropy()),
        "weighted_null": weighted,
        "note": "DC-SBM null already contains block structure, so this is a harder, "
                "more conservative test than the configuration-model null.",
    }
    if len(a):
        sd = float(a.std(ddof=1)) if len(a) > 1 else 0.0
        out["dcsbm_null"] = {"null_mean": float(a.mean()), "null_std": sd,
                             "z": float((Q_obs - a.mean()) / sd) if sd > 0 else None,
                             "emp_p": float((a >= Q_obs).mean()), "n": int(len(a))}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--start", default="2026-01-28")
    ap.add_argument("--end", default="2026-02-20")
    ap.add_argument("--exclude-dates", default=None)
    ap.add_argument("--n-null", type=int, default=100)
    ap.add_argument("--n-jobs", type=int, default=-1)
    ap.add_argument("--out", default="results/dcsbm_null.json")
    args = ap.parse_args()

    excl = [d.strip() for d in args.exclude_dates.split(",")] if args.exclude_dates else None
    posts, comments = R.prepare(args.data_dir, args.start, args.end, None, excl)
    g_ig = R.build_graph(R.build_edges(posts, comments))

    res = {"window": [args.start, args.end]}
    res["topology_null"] = R._safe(lambda: dcsbm_null(g_ig, args.n_null, args.n_jobs, weighted=False))
    res["weighted_null"] = R._safe(lambda: dcsbm_null(g_ig, args.n_null, args.n_jobs, weighted=True))

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(res, f, indent=2, default=str)
    print(json.dumps(res, indent=2, default=str))


if __name__ == "__main__":
    main()
