"""
export_openings.py — real job openings and a market snapshot for every occupation → data/openings.json

Uses the same title matching (OCC patterns) and skill lexicon as kg_build.py. This is an application layer for the
website: it does not change the knowledge graph, the models or any evaluation metric.
Openings are LinkedIn postings from the Kaggle dataset (2024), so links may have expired.

Run:  python pipeline/src/export_openings.py raw
"""
import json, re, sys
from collections import Counter
from pathlib import Path
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
from kg_build import OCC, CONCEPTS, AMBIG, norm

ROOT = HERE.parent.parent; KG = HERE.parent / "kg"
RAW = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "raw"
PER_OCC = 60                                     # openings shown per occupation (most recent first)

skills = pd.read_csv(KG / "nodes_skill.csv")
names = list(skills.id); kinds = dict(zip(skills.id, skills.kind)); sidx = {n: i for i, n in enumerate(names)}
conc = {c: re.compile(p, re.I) for c, p in CONCEPTS.items() if c in sidx}
amb = {c: re.compile(p) for c, p in AMBIG.items() if c in sidx}
keys = {n: norm(n) for n in names if kinds[n] == "tool" and n not in amb}

def extract(raw):
    nt = norm(raw); out = [sidx[n] for n, k in keys.items() if k in nt]
    out += [sidx[c] for c, rx in conc.items() if rx.search(nt)]
    out += [sidx[c] for c, rx in amb.items() if rx.search(raw)]
    return sorted(set(out))

cols = ["job_id", "company_name", "title", "description", "skills_desc", "location", "normalized_salary", "remote_allowed",
        "formatted_experience_level", "formatted_work_type", "job_posting_url", "listed_time"]
P = pd.read_csv(RAW / "postings.csv", usecols=cols)
lab = pd.Series([None] * len(P), index=P.index, dtype=object)
for code, name, dom, pat in OCC:                 # same first-match rule as kg_build.py
    m = lab.isna() & P["title"].str.contains(pat, case=False, regex=True, na=False)
    lab[m] = code
P["occ"] = lab; P = P[P["occ"].notna()].copy()
P.loc[~P.normalized_salary.between(15_000, 600_000), "normalized_salary"] = np.nan

def state(loc):                                  # "Austin, TX" → "TX"; "United States" stays as is
    loc = str(loc).strip(); m = re.search(r",\s*([A-Z]{2})$", loc)
    return m.group(1) if m else loc

P["state"] = P.location.map(state)
occ_codes = set(pd.read_csv(KG / "nodes_occupation.csv").id)
out = {"note": "LinkedIn job postings from the Kaggle dataset (2024). Links may have expired.", "occupations": {}}
for code, d in P.groupby("occ"):
    if code not in occ_codes: continue
    sal = d.normalized_salary.dropna()
    stats = dict(postings=int(len(d)),
                 salary_median=int(sal.median()) if len(sal) else None,
                 salary_p25=int(sal.quantile(.25)) if len(sal) else None,
                 salary_p75=int(sal.quantile(.75)) if len(sal) else None,
                 salary_n=int(len(sal)),
                 remote_pct=round(100 * float(d.remote_allowed.fillna(0).astype(bool).mean()), 1),
                 top_locations=[[k, int(v)] for k, v in d.location.value_counts().head(6).items()],
                 top_states=[[k, int(v)] for k, v in d.state.value_counts().head(8).items()],
                 top_companies=[[k, int(v)] for k, v in d.company_name.value_counts().head(6).items()],
                 experience={k: int(v) for k, v in d.formatted_experience_level.value_counts().items()})
    recent = d.sort_values("listed_time", ascending=False).head(PER_OCC)
    jobs = []
    for r in recent.itertuples():
        raw = f"{r.title}\n{r.description if isinstance(r.description, str) else ''}\n{r.skills_desc if isinstance(r.skills_desc, str) else ''}"
        jobs.append(dict(id=int(r.job_id), title=r.title, company=r.company_name if isinstance(r.company_name, str) else "Company not listed",
                         location=r.location, state=r.state, remote=bool(r.remote_allowed == 1),
                         salary=None if pd.isna(r.normalized_salary) else int(r.normalized_salary),
                         experience=r.formatted_experience_level if isinstance(r.formatted_experience_level, str) else "",
                         work_type=r.formatted_work_type if isinstance(r.formatted_work_type, str) else "",
                         date=str(pd.to_datetime(r.listed_time, unit="ms").date()), url=r.job_posting_url, skills=extract(raw)))
    out["occupations"][code] = dict(stats=stats, jobs=jobs)
(ROOT / "data").mkdir(exist_ok=True)
json.dump(out, open(ROOT / "data" / "openings.json", "w"), separators=(",", ":"))
n = sum(len(v["jobs"]) for v in out["occupations"].values())
print(f"openings.json: {len(out['occupations'])} occupations, {n} openings, {(ROOT/'data'/'openings.json').stat().st_size/1e6:.2f} MB")
