"""
kg_experiments.py — the two experiments that test the project's novelty.

Experiment 1 · Edge-weighting ablation
    Same models, same test sets, five ways of weighting occupation→skill edges:
      uniform          every edge = 1                       (no weighting at all)
      confidence_only  w = c                                 (two-source agreement only)
      demand_only      w = 0.3 + 0.7·d/d_max                 (freshness-weighted demand only)
      no_freshness     w = c·(0.3 + 0.7·d_raw/d_raw,max)      (demand without posting-freshness damping)
      full             w = c·(0.3 + 0.7·d/d_max)             (the proposed weighting)
    Weights enter the models through Node2Vec's walks, PREPARES_FOR, the HGT loss and the skill-overlap scorer.
    The evaluation protocol is identical for every variant: same edge splits, same simulated profiles
    (always sampled with the full weights) and the same real held-out postings.

Experiment 2 · Standard recommenders ("normal recommendation")
    Occupation recommendation: Random, Popularity (most-posted occupations), Jaccard skill overlap,
    TF-IDF content-based matching, weighted skill overlap.
    Link prediction (occupation→skill): Popularity, item-based and user-based collaborative filtering.
    Same seeds, splits and profiles as kg_models.py, so the numbers are directly comparable with
    Node2Vec and HGT in results/metrics.json.

Run:  python pipeline/src/kg_experiments.py            (HGT parts run only if PyTorch is installed)
Out:  pipeline/results/experiments.json  and  pipeline/results/experiments.md
"""
import json, sys, time
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
import kg_models as km                                                 # noqa: E402

SEEDS = [int(s) for s in (sys.argv[1].split(",") if len(sys.argv) > 1 and not sys.argv[1].startswith("--") else ["0", "1", "2"])]
RES = HERE.parent / "results"
REQ = km.REQ
o_ix = [km.oid[o] for o in REQ.occ]; s_ix = [km.sid[s] for s in REQ.skill]
dmax = REQ.groupby("occ").demand.transform("max").replace(0, 1)
draw = REQ.mentions / REQ.n_postings.clip(lower=1); draw_max = draw.groupby(REQ.occ).transform("max").replace(0, 1)
VARIANTS = {
    "uniform": np.ones(len(REQ)),
    "confidence_only": REQ.confidence.values,
    "demand_only": (0.3 + 0.7 * REQ.demand / dmax).values,
    "no_freshness": (REQ.confidence * (0.3 + 0.7 * draw / draw_max)).values,
    "full": REQ.weight.values,
}
WMAP = {v: {(o, s): float(w) for o, s, w in zip(o_ix, s_ix, ws)} for v, ws in VARIANTS.items()}
POSTINGS = np.zeros(km.nO)
for o, n in zip(o_ix, REQ.n_postings): POSTINGS[o] = max(POSTINGS[o], n)

def reweight(edges, v):
    return [(o, s, WMAP[v][(o, s)]) for o, s, _ in edges]

def rec_eval(fn, sim, held):
    return {"simulated": km.rec_metrics(km.occ_ranks(sim, fn)), "heldout_postings": km.rec_metrics(km.occ_ranks(held, fn))}

def lp_eval(M, test, seed):
    r = km.rank_metrics(test, lambda u: M[u], km.req_pos)
    r["AUC"] = km.auc(test, lambda u, v: M[u, v] + 1e-9 * np.random.rand(), km.req_pos, km.nS, seed)
    return r

def run(seed):
    t0 = time.time(); out = {"seed": seed}
    req_tr, req_va, req_te, tea_tr, tea_va, tea_te = km.split_edges(seed)
    sim = km.simulated_profiles(req_tr, seed); held = km.heldout_profiles(seed)
    B = np.zeros((km.nO, km.nS))
    for o, s, _ in req_tr: B[o, s] = 1.0

    # ---------------- Experiment 2: standard recommenders
    rng = np.random.default_rng(seed)
    df = B.sum(0); idf = np.log((1 + km.nO) / (1 + df)) + 1
    T = B * idf; T = T / (np.linalg.norm(T, axis=1, keepdims=True) + 1e-9)
    def tfidf(sk):
        u = np.zeros(km.nS); u[sk] = idf[sk]; u /= (np.linalg.norm(u) + 1e-9); return T @ u
    def jaccard(sk):
        u = np.zeros(km.nS); u[sk] = 1; inter = B @ u; return inter / (B.sum(1) + len(set(sk)) - inter + 1e-9)
    Wf = np.zeros((km.nO, km.nS))
    for o, s, w in req_tr: Wf[o, s] = w
    def overlap(sk): return Wf[:, sk].sum(1) / Wf.sum(1)
    base = {"Random": lambda sk: rng.random(km.nO), "Popularity": lambda sk: POSTINGS + 1e-6 * rng.random(km.nO),
            "Jaccard overlap": jaccard, "TF-IDF content-based": tfidf, "Weighted skill overlap": overlap}
    out["exp2_recommendation"] = {name: rec_eval(fn, sim, held) for name, fn in base.items()}
    deg = B.sum(0); POP = np.tile(deg, (km.nO, 1))
    nb = B / (np.linalg.norm(B, axis=0, keepdims=True) + 1e-9); Ssk = nb.T @ nb; np.fill_diagonal(Ssk, 0); ICF = B @ Ssk
    nu = B / (np.linalg.norm(B, axis=1, keepdims=True) + 1e-9); Soc = nu @ nu.T; np.fill_diagonal(Soc, 0); UCF = Soc @ B
    out["exp2_link_prediction"] = {"Popularity": lp_eval(POP, req_te, seed), "Item-based CF": lp_eval(ICF, req_te, seed),
                                   "User-based CF": lp_eval(UCF, req_te, seed)}

    # ---------------- Experiment 1: weighting ablation
    out["exp1"] = {}
    for v in VARIANTS:
        rtr = reweight(req_tr, v); prep = km.prepares_from(rtr, tea_tr)
        Eo, Es, Ec = km.node2vec(rtr, tea_tr, km.rel_all, prep, seed)
        N2V = km.cos(Eo, Es)
        W = np.zeros((km.nO, km.nS))
        for o, s, w in rtr: W[o, s] = w
        res = {"Node2Vec": {"LP_REQUIRES": lp_eval(N2V, req_te, seed), **rec_eval(lambda sk: N2V[:, sk].mean(1), sim, held)},
               "Weighted skill overlap": rec_eval(lambda sk, W=W: W[:, sk].sum(1) / W.sum(1), sim, held)}
        if km.HAS_TORCH:
            data = km.hetero(rtr, tea_tr, km.rel_all, prep, Eo, Es, Ec)
            model, h = km.train_hgt(data, rtr, tea_tr, req_va, seed)
            with km.torch.no_grad():
                H = km.torch.stack([model.score(h, "requires", "occupation", km.torch.full((km.nS,), o), km.torch.arange(km.nS)) for o in range(km.nO)]).numpy()
            res["HGT"] = {"LP_REQUIRES": lp_eval(H, req_te, seed), **rec_eval(lambda sk: (1 / (1 + np.exp(-H[:, sk]))).mean(1), sim, held)}
        out["exp1"][v] = res
        print(f"  seed {seed} · {v:<16} done", flush=True)
    out["seconds"] = round(time.time() - t0, 1)
    return out

def agg(vals):
    return {"mean": float(np.mean(vals)), "sd": float(np.std(vals))}

def summarise(runs):
    def walk(objs):
        if isinstance(objs[0], dict): return {k: walk([o[k] for o in objs]) for k in objs[0]}
        return agg(objs)
    keep = [{k: r[k] for k in ("exp1", "exp2_recommendation", "exp2_link_prediction")} for r in runs]
    return walk(keep)

def markdown(S):
    f = lambda d: f"{d['mean']:.3f} ± {d['sd']:.3f}"
    L = ["# Experiment results (mean ± sd over seeds " + ", ".join(map(str, SEEDS)) + ")", "",
         "## Experiment 2a · Occupation recommendation vs standard recommenders", "",
         "| Method | MRR (simulated) | MRR (real held-out) | Recall@10 (real held-out) | NDCG@10 (real held-out) |", "|---|---|---|---|---|"]
    for m, d in S["exp2_recommendation"].items():
        L.append(f"| {m} | {f(d['simulated']['MRR'])} | {f(d['heldout_postings']['MRR'])} | {f(d['heldout_postings']['Recall@10'])} | {f(d['heldout_postings']['NDCG@10'])} |")
    for m in ("Node2Vec", "HGT"):
        if m in S["exp1"]["full"]:
            d = S["exp1"]["full"][m]
            L.append(f"| **{m} (ours)** | {f(d['simulated']['MRR'])} | {f(d['heldout_postings']['MRR'])} | {f(d['heldout_postings']['Recall@10'])} | {f(d['heldout_postings']['NDCG@10'])} |")
    L += ["", "## Experiment 2b · Link prediction (occupation → skill) vs collaborative filtering", "", "| Method | AUC | Hits@10 | MRR |", "|---|---|---|---|"]
    for m, d in S["exp2_link_prediction"].items():
        L.append(f"| {m} | {f(d['AUC'])} | {f(d['Hits@10'])} | {f(d['MRR'])} |")
    for m in ("Node2Vec", "HGT"):
        if m in S["exp1"]["full"]:
            d = S["exp1"]["full"][m]["LP_REQUIRES"]; L.append(f"| **{m} (ours)** | {f(d['AUC'])} | {f(d['Hits@10'])} | {f(d['MRR'])} |")
    L += ["", "## Experiment 1 · Edge-weighting ablation", ""]
    models = [m for m in ("Node2Vec", "HGT", "Weighted skill overlap") if m in S["exp1"]["full"]]
    for m in models:
        L += [f"### {m}", "", "| Weighting | MRR (simulated) | MRR (real held-out) | Recall@10 (real held-out) |" + (" LP AUC | LP MRR |" if m != "Weighted skill overlap" else ""),
              "|---|---|---|---|" + ("---|---|" if m != "Weighted skill overlap" else "")]
        for v in VARIANTS:
            d = S["exp1"][v][m]
            row = f"| {v} | {f(d['simulated']['MRR'])} | {f(d['heldout_postings']['MRR'])} | {f(d['heldout_postings']['Recall@10'])} |"
            if m != "Weighted skill overlap": row += f" {f(d['LP_REQUIRES']['AUC'])} | {f(d['LP_REQUIRES']['MRR'])} |"
            L.append(row)
        L.append("")
    if not km.HAS_TORCH:
        L.append("_HGT rows are missing because PyTorch is not installed; run this script on a machine with PyTorch to add them._")
    return "\n".join(L)

def finish(runs):
    global SEEDS
    SEEDS = [r["seed"] for r in runs]
    S = summarise(runs)
    json.dump({"seeds": SEEDS, "hgt_included": km.HAS_TORCH, "summary": S, "runs": runs}, open(RES / "experiments.json", "w"), indent=1)
    md = markdown(S); (RES / "experiments.md").write_text(md, encoding="utf-8"); print("\n" + md)

if __name__ == "__main__":
    # python kg_experiments.py 0,1,2        → run all seeds and summarise
    # python kg_experiments.py 0 --save     → run one seed and save results/exp_run_0.json (for short time limits)
    # python kg_experiments.py --aggregate  → summarise every saved results/exp_run_*.json
    if "--aggregate" in sys.argv:
        finish([json.load(open(f)) for f in sorted(RES.glob("exp_run_*.json"))])
    else:
        print(f"Running experiments for seeds {SEEDS} (HGT {'on' if km.HAS_TORCH else 'off: PyTorch not installed'}) …", flush=True)
        runs = []
        for sd in SEEDS:
            r = run(sd); runs.append(r)
            if "--save" in sys.argv: json.dump(r, open(RES / f"exp_run_{sd}.json", "w"))
        if "--save" not in sys.argv: finish(runs)
