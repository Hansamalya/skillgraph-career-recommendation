"""
kg_paper_figures.py — regenerates the IEEE paper's result figures from the results files (no hand-typed numbers).

  Fig. 2  figures/paper_fig2_results.png     (a) occupation→skill link prediction vs popularity / CF
                                             (b) occupation recommendation on real held-out postings vs standard recommenders
  Fig. 3  figures/paper_fig3_case_study.png  explanation for a learner with Python and SQL targeting Data Scientist

Inputs: results/metrics.json (kg_models.py), results/experiments.json (kg_experiments.py), ../data/kg_web.json (export_web.py)
Run:    python pipeline/src/kg_paper_figures.py [--skills Python SQL] [--occupation 15-2051.00]
"""
import argparse, json, textwrap
from pathlib import Path
import logging, matplotlib
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

P = Path(__file__).resolve().parent.parent; RES = P / "results"; OUT = P / "figures"; OUT.mkdir(exist_ok=True)
plt.rcParams.update({"font.family": ["Liberation Serif", "Times New Roman", "DejaVu Serif"], "font.size": 7,
                     "axes.spines.top": False, "axes.spines.right": False})
COL = {"Popularity": "#BFBFBF", "User-based CF": "#9DB3D6", "Node2Vec": "#E2B64B", "HGT": "#7B61B8", "Random": "#E3E3E3",
       "TF-IDF": "#8FBF9F", "Skill overlap": "#C7B9E8"}

def fig2():
    M = json.load(open(RES / "metrics.json"))["summary"]; X = json.load(open(RES / "experiments.json"))["summary"]
    L = X["exp2_link_prediction"]; R = X["exp2_recommendation"]
    lp = {"Popularity": L["Popularity"], "User-based CF": L["User-based CF"], "Node2Vec": M["LP/REQUIRES/Node2Vec"], "HGT": M["LP/REQUIRES/HGT"]}
    rec = {"Random": R["Random"]["heldout_postings"], "Popularity": R["Popularity"]["heldout_postings"], "TF-IDF": R["TF-IDF content-based"]["heldout_postings"],
           "Skill overlap": R["Weighted skill overlap"]["heldout_postings"], "Node2Vec": M["REC/heldout_postings/Node2Vec"], "HGT": M["REC/heldout_postings/HGT"]}
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(3.45, 3.9))
    for i, (m, d) in enumerate(lp.items()):
        a1.bar([0 + (i - 1.5) * .2, 1 + (i - 1.5) * .2], [d["AUC"]["mean"], d["MRR"]["mean"]], .2, yerr=[d["AUC"]["sd"], d["MRR"]["sd"]],
               color=COL[m], label=m, capsize=1.5, error_kw={"lw": .6})
    a1.set_xticks([0, 1]); a1.set_xticklabels(["AUC", "MRR"]); a1.set_ylim(0, 1.15); a1.set_yticks([0, .25, .5, .75, 1])
    a1.set_title("(a) Occupation→skill link prediction (held-out edges)", fontsize=7.5); a1.legend(fontsize=6, frameon=False, ncol=4, loc="upper center")
    for i, (m, d) in enumerate(rec.items()):
        a2.bar([0 + (i - 2.5) * .135, 1 + (i - 2.5) * .135], [d["MRR"]["mean"], d["Recall@10"]["mean"]], .135, color=COL[m], label=m)
    n = json.load(open(RES / "metrics.json"))["runs"][0]["n_profiles/heldout_postings"]
    a2.set_xticks([0, 1]); a2.set_xticklabels(["MRR", "Recall@10"]); a2.set_ylim(0, 1.15); a2.set_yticks([0, .25, .5, .75, 1])
    a2.set_title(f"(b) Occupation recommendation, {n} real held-out postings", fontsize=7.5); a2.legend(fontsize=6, frameon=False, ncol=3, loc="upper left")
    fig.tight_layout(h_pad=1.2); fig.savefig(OUT / "paper_fig2_results.png", dpi=300); plt.close(fig)

def fig3(skills, occ_id):
    D = json.load(open(P.parent / "data" / "kg_web.json"))
    SK = [s["name"] for s in D["skills"]]; occ = next(o for o in D["occupations"] if o["id"] == occ_id)
    req = {e[0]: e for e in occ["requires"]}; have = [SK.index(s) for s in skills if s in SK and SK.index(s) in req]
    teach = {}
    for c in D["courses"]:
        for s in c["teaches"]: teach.setdefault(s, c)
    missing = [e for e in sorted(occ["requires"], key=lambda e: -e[1]) if e[0] not in have and e[0] in teach][:4]
    fig, ax = plt.subplots(figsize=(3.45, 3.25)); ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.axis("off")
    def box(x, y, w, h, t, fc, ec, fs=6.3, bold=False):
        ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0.3,rounding_size=1.5", fc=fc, ec=ec, lw=.8))
        ax.text(x, y, t, ha="center", va="center", fontsize=fs, fontweight="bold" if bold else "normal", linespacing=1.1)
    def arrow(x1, y1, x2, y2, col, dashed=False):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1), arrowprops=dict(arrowstyle="-|>", color=col, lw=.8, mutation_scale=6, ls="--" if dashed else "-"))
    CX, SX, OX, OY = 15, 50, 88, 55
    box(CX, 88, 26, 9, "Learner\n(" + ", ".join(SK[s] for s in have) + ")", "#DCE8F7", "#7DA7D9", bold=True)
    box(OX, OY, 22, 12, "\n".join(textwrap.wrap(occ["name"], 12)), "#DDF0DA", "#2E9E4F", fs=6.8, bold=True)
    for s, y in zip(have, [92, 82]):
        box(SX, y, 24, 6.5, SK[s], "#D6EBE8", "#1F7A74"); arrow(28, 88, SX - 12, y, "#1F7A74"); arrow(SX + 12, y, OX - 11, OY + 4, "#1F7A74")
        ax.text(SX + 13.5, y + 3.2, f"w {req[s][1]:.2f}", fontsize=5.4, color="#1F7A74")
    ax.text(32, 93.5, "HAS", fontsize=5.4, color="#1F7A74")
    for e, y in zip(missing, [64, 51, 38, 25]):
        c = textwrap.fill(teach[e[0]]["name"], 22)
        box(SX, y, 24, 6.5, SK[e[0]], "#F7DDE0", "#D1495B"); arrow(SX + 12, y, OX - 11, OY - 3, "#D1495B", dashed=True)
        ax.text(SX + 13.5, y + 3.2, f"w {e[1]:.2f}", fontsize=5.4, color="#D1495B")
        box(CX, y, 28, 9.5 if "\n" in c else 6.5, c, "#E3EAF5", "#6C8EAD", fs=5.1); arrow(CX + 14, y, SX - 12, y, "#6C8EAD")
    ax.text(SX, 99, "matched skills", ha="center", fontsize=5.8, color="#1F7A74", fontstyle="italic")
    ax.text(SX, 71.5, "missing skills (ranked by w)", ha="center", fontsize=5.8, color="#D1495B", fontstyle="italic")
    ax.text(CX, 71.5, "courses (TEACHES)", ha="center", fontsize=5.8, color="#6C8EAD", fontstyle="italic")
    ax.text(50, 12, "solid: REQUIRES (matched) · dashed: REQUIRES (missing)\nw = edge weight (demand × confidence)", ha="center", fontsize=5.2, color="#444", linespacing=1.2)
    fig.savefig(OUT / "paper_fig3_case_study.png", dpi=300, bbox_inches="tight", pad_inches=.03); plt.close(fig)
    return [(SK[s], req[s][1]) for s in have], [(SK[e[0]], e[1], teach[e[0]]["name"]) for e in missing]

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--skills", nargs="+", default=["Python", "SQL"]); ap.add_argument("--occupation", default="15-2051.00")
    a = ap.parse_args()
    fig2(); matched, missing = fig3(a.skills, a.occupation)
    print("Fig. 2 →", OUT / "paper_fig2_results.png"); print("Fig. 3 →", OUT / "paper_fig3_case_study.png")
    print("  matched:", matched); print("  missing → course:", missing)
