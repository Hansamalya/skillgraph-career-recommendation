"""
SkillGraph — Streamlit version.
Knowledge Graph Embedding on Skill and Career Recommendation (Hansamalya B · Moniisha S · Guide: Dr. Manivannan A, VIT Chennai)

Run:  streamlit run streamlit_app.py        (VS Code: Run and Debug ▸ "▶ Run SkillGraph (Streamlit)")
Uses the same engine (backend/kg_engine.py), data (data/kg_web.json, data/openings.json) and live-jobs module as the Flask site.
"""
import io, json, math, os, sys
from pathlib import Path
import numpy as np, pandas as pd, networkx as nx, streamlit as st
import plotly.graph_objects as go

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "backend"))
from kg_engine import Engine                                   # noqa: E402
import live_jobs                                               # noqa: E402

def _load_env(path):                                           # reads ADZUNA_APP_ID / ADZUNA_APP_KEY from .env
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if "=" in line and not line.strip().startswith("#"):
                k, v = line.split("=", 1); os.environ.setdefault(k.strip(), v.strip().strip('"'))
_load_env(ROOT / ".env")

ASSETS = ROOT / "assets"
st.set_page_config(page_title="SkillGraph", page_icon=str(ASSETS / "favicon.png"), layout="wide")
st.logo(str(ASSETS / "logo_wordmark.png"), icon_image=str(ASSETS / "logo.png"), size="large")
DOMAIN_COL = {"AI & Machine Learning": "#2E9E4F", "Data Analytics & BI": "#1F77B4", "Cloud & DevOps": "#8C564B",
              "Software & Web": "#9467BD", "Cybersecurity": "#D62728", "UI/UX & Design": "#C2549D"}
MODELS = {"hgt": "HGT", "node2vec": "Node2Vec", "overlap": "Skill overlap"}

@st.cache_resource
def engine():
    return Engine()

@st.cache_data
def openings_data():
    p = ROOT / "data" / "openings.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {"occupations": {}, "note": ""}

E = engine(); DB = E.db
SK = [s["name"] for s in DB["skills"]]
OCC = DB["occupations"]; OCC_BY_ID = {o["id"]: (i, o) for i, o in enumerate(OCC)}
OCC_COUNT = [sum(1 for r in E.req if s in r and r[s]["w"] >= 0.3) for s in range(len(SK))]

# ------------------------------------------------------------------ state + styling
ss = st.session_state
ss.setdefault("skills", []); ss.setdefault("name", ""); ss.setdefault("model", "hgt"); ss.setdefault("progress", {})
st.markdown("""<style>
.chip{display:inline-block;padding:3px 10px;margin:2px 3px 2px 0;border-radius:999px;font-size:.85rem;background:#eef2f0;border:1px solid #d5dfda}
.chip.have{background:#d6ebe8;border-color:transparent;color:#0f4a46}.chip.gap{background:#f7dde0;border-color:transparent;color:#8a2433}
.chip.sky{background:#dfe7ef;border-color:transparent;color:#2c4a66}
.kgpath{padding:8px 12px;border-left:3px solid #f2a541;background:#f8faf9;border-radius:8px;margin:6px 0;font-size:.92rem}
.kgpath.rel{border-left-color:#1f7a74}
.jobcard{padding:12px 14px;border:1px solid #e3e9e6;border-radius:12px;background:#fff;margin-bottom:10px}
.jobcard h4{margin:0 0 4px 0;font-size:1rem}.jobmeta{color:#5b7083;font-size:.85rem}
.bar{height:6px;border-radius:9px;background:#eef2f0;overflow:hidden;margin:8px 0 4px}.bar i{display:block;height:100%;background:#f2a541}
.dchip{display:inline-block;padding:3px 10px;border-radius:999px;font-size:.8rem;font-weight:600;color:#fff}
</style>""", unsafe_allow_html=True)

def dslug(d):
    import re as _re
    return _re.sub(r"[^a-z0-9]+", "-", d.lower()).strip("-")

def occ_img(o):
    """Real photo if fetch_images.py downloaded it, otherwise the built-in banner for the occupation's domain."""
    ph = ASSETS / "img" / f"{o['id']}.jpg"
    return str(ph if ph.exists() else ASSETS / "banners" / f"{dslug(o['domain'])}.jpg")

def hero_img():
    ph = ASSETS / "img" / "hero.jpg"
    return str(ph if ph.exists() else ASSETS / "banners" / "hero.jpg")

def chips(names, cls=""):
    return " ".join(f'<span class="chip {cls}">{n}</span>' for n in names) or "<span style='color:#5b7083'>none</span>"
def dchip(d):
    return f'<span class="dchip" style="background:{DOMAIN_COL[d]}">{d}</span>'
def U():
    return E.skills(ss.skills)

def ranked(model=None):
    return E.rank(ss.skills, model or ss.model)

def score_text(r, model):
    return f"{round(r['score'] * 100)}%" if model == "overlap" else f"{r['score']:.3f}"

# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.caption("Knowledge Graph Embedding on Skill and Career Recommendation")
    ss.name = st.text_input("Your name", ss.name)
    ss.model = st.radio("Rank careers with", list(MODELS), format_func=MODELS.get, index=list(MODELS).index(ss.model),
                        help="HGT: mean HGT link score · Node2Vec: mean cosine similarity · Skill overlap: share of weighted skills you have")
    st.metric("Skills on your profile", len(U()))
    st.caption("Profile is kept for this browser session.")

# ================================================================== pages
def home():
    st.image(hero_img(), use_container_width=True)
    st.title("See which careers your skills connect to")
    st.write("SkillGraph links **27 occupations, 475 skills and 229 courses** in one knowledge graph, ranks careers with a "
             "Heterogeneous Graph Transformer, and shows the skills you have, the skills you're missing and the course that closes each gap.")
    s = DB["stats"]; S = DB["summary"]
    c = st.columns(4)
    c[0].metric("Graph nodes", f"{s['nodes']['total']:,}"); c[1].metric("Graph edges", f"{s['edges']['total']:,}")
    c[2].metric("LinkedIn postings matched", f"{s['postings']['matched']:,}"); c[3].metric("HGT link-prediction AUC", f"{S['LP/REQUIRES/HGT']['AUC']['mean']:.3f}")
    st.subheader("How it works")
    steps = [("Build the graph", "O*NET occupations, LinkedIn postings and Coursera courses become typed nodes and edges."),
             ("Weight every edge", "Fresher postings count slightly more; skills confirmed by both sources get higher confidence."),
             ("Learn embeddings", "Node2Vec captures structure; a two-layer HGT refines it."),
             ("Map your skills", "Your résumé is read and matched to skill nodes."),
             ("Rank and explain", "Each career comes with matched skills, missing skills, graph paths, courses and openings.")]
    cols = st.columns(5)
    for i, (h, t) in enumerate(steps):
        cols[i].markdown(f"**{i + 1}. {h}**  \n<span style='color:#5b7083;font-size:.9rem'>{t}</span>", unsafe_allow_html=True)
    st.subheader("27 occupations across six domains")
    doms = list(DOMAIN_COL)
    for row in range(0, len(doms), 3):
        cols = st.columns(3)
        for col, d in zip(cols, doms[row:row + 3]):
            with col:
                st.image(str(ASSETS / "banners" / f"{dslug(d)}_titled.jpg"), use_container_width=True)
                st.markdown(chips([o["name"] for o in OCC if o["domain"] == d]), unsafe_allow_html=True)
    st.info("Start on **Profile**: upload a résumé or pick your skills.", icon="👉")

def profile():
    st.title("Your profile")
    left, right = st.columns([1.15, 1])
    with left:
        st.subheader("📄 Résumé upload")
        f = st.file_uploader("PDF, DOCX or TXT", type=["pdf", "docx", "txt", "md"])
        if f is not None:
            data = f.getvalue(); name = f.name.lower()
            try:
                if name.endswith(".pdf"):
                    from pypdf import PdfReader; text = "\n".join(p.extract_text() or "" for p in PdfReader(io.BytesIO(data)).pages)
                elif name.endswith(".docx"):
                    import docx; text = "\n".join(p.text for p in docx.Document(io.BytesIO(data)).paragraphs)
                else:
                    text = data.decode("utf-8", "ignore")
                found = E.extract(text); new = [s for s in found if s not in ss.skills]
                st.markdown(f"**{len(found)} graph skills found** in {f.name}")
                st.markdown(chips(found, "have"), unsafe_allow_html=True)
                if new:
                    st.button(f"Add {len(new)} skill(s) to my profile", type="primary",
                              on_click=lambda n=new: ss.update(skills=ss.skills + [s for s in n if s not in ss.skills],
                                                               skills_ms=ss.skills + [s for s in n if s not in ss.skills]))
            except Exception as e:
                st.error(f"Couldn't read that file ({e}). Upload a text-based PDF, DOCX or TXT.")
        st.subheader("Your skills")
        # kept in ss.skills (persists across pages); the widget has its own key, synced on change
        st.multiselect("Skills from the knowledge graph (type to search)", options=sorted(SK, key=str.lower), default=ss.skills,
                       key="skills_ms", on_change=lambda: ss.update(skills=list(ss.skills_ms)))
    with right:
        st.subheader("🎯 Top matches")
        if not U():
            st.write("Add skills to see your matches.")
        else:
            for r in ranked()[:4]:
                a, b = st.columns([1, 3])
                a.image(occ_img(OCC_BY_ID[r["id"]][1]), use_container_width=True)
                with b:
                    st.markdown(f"**{r['name']}** &nbsp; {dchip(r['domain'])}", unsafe_allow_html=True)
                    st.progress(min(1.0, r["coverage"]), text=f"{MODELS[ss.model]} score {score_text(r, ss.model)} · covers {round(r['coverage'] * 100)}% of weighted skills")
        st.subheader("Skills required by the most occupations")
        top = sorted(range(len(SK)), key=lambda i: -OCC_COUNT[i])[:12]; have = set(U())
        fig = go.Figure(go.Bar(x=[OCC_COUNT[i] for i in top][::-1], y=[SK[i] for i in top][::-1], orientation="h",
                               marker_color=["#1f7a74" if i in have else "#c9d5d0" for i in top][::-1]))
        fig.update_layout(height=360, margin=dict(l=0, r=10, t=10, b=10), xaxis_title="occupations (weight ≥ 0.3)")
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Teal = on your profile.")

def gap_table(oi, n=12):
    have = set(U())
    rows = sorted(E.req[oi].items(), key=lambda kv: -kv[1]["w"])[:n]
    return pd.DataFrame([{"Skill": SK[s], "You have": "✅" if s in have else "❌", "Weight": e["w"],
                          "Demand (% of postings)": round(100 * e["d"], 1), "Confidence": e["c"], "Source": e["src"]} for s, e in rows])

def score_jobs(jobs, oi):
    have = set(U()); out = []
    for j in jobs:
        sk = [s if isinstance(s, int) else E.skill_index.get(str(s).lower(), -1) for s in j["skills"]]
        sk = [s for s in sk if s >= 0]; h = [s for s in sk if s in have]
        miss = sorted([s for s in sk if s not in have], key=lambda s: -E.req[oi].get(s, {"w": 0})["w"])
        out.append({**j, "sk": sk, "have": h, "miss": miss, "match": len(h) / len(sk) if sk else 0, "rank": len(h) / (len(sk) + 2)})
    out.sort(key=lambda j: str(j.get("date", "")), reverse=True)        # newest first …
    return sorted(out, key=lambda j: -j["rank"])                          # … then best match (stable sort keeps date order)

def job_cards(jobs, limit=10):
    if not jobs:
        st.write("No openings found for this search."); return
    cols = st.columns(2)
    for k, j in enumerate(jobs[:limit]):
        sal = f"${round(j['salary'] / 1000)}k" if isinstance(j.get("salary"), (int, float)) and j["salary"] else (j.get("salary") or "")
        miss = chips([SK[s] for s in j["miss"][:4]], "gap") if j["miss"] else ""
        cols[k % 2].markdown(f"""<div class="jobcard"><h4><a href="{j['url']}" target="_blank">{j['title']}</a></h4>
          <div class="jobmeta">{j['company']} · {j['location']}{' · Remote' if j.get('remote') else ''}{(' · ' + j['experience']) if j.get('experience') else ''}{(' · ' + j['date']) if j.get('date') else ''}{(' · <b style="color:#1f7a74">' + sal + '</b>') if sal else ''}</div>
          <div class="bar"><i style="width:{round(j['match'] * 100)}%"></i></div>
          <div class="jobmeta">{f"You have {len(j['have'])} of {len(j['sk'])} skills in this posting" if j['sk'] else "No graph skills found in this posting"}</div>{miss}</div>""",
          unsafe_allow_html=True)

def openings_section(occ, oi):
    st.subheader("💼 Job openings")
    t1, t2, t3 = st.tabs(["Live 2026 · Adzuna", "Remote now · Remotive", "2024 dataset"])
    with t1:
        L = live_jobs.LOCATIONS
        c1, c2, c3 = st.columns([1, 1.2, .6])
        country = c1.selectbox("Country", list(L), key="cty", format_func=lambda k: L[k]["name"])
        labels = {v: l for v, l in L[country]["cities"]}
        options = [""] + list(labels) + ["__other"]
        city = c2.selectbox("City", options, key=f"city_{country}", index=1 if len(options) > 2 else 0,
                            format_func=lambda v: f"All of {L[country]['name']}" if v == "" else "Other city…" if v == "__other" else labels[v])
        where = st.text_input("Type a city", key=f"other_{country}").strip() if city == "__other" else city
        if c3.button("Search", key="liveGo", use_container_width=True) or ss.get("live_key") == (occ["id"], country, where):
            ss.live_key = (occ["id"], country, where)
            try:
                with st.spinner("Searching current openings…"):
                    res = live_jobs.adzuna(occ["id"], country, where)
                jobs = [{**j, "skills": E.extract(j.pop("text"))} for j in [dict(x) for x in res["jobs"]]]
                if res.get("note"): st.info(res["note"])
                st.caption(f"{len(jobs)} current openings for “{occ['name']}”{(' near ' + where) if where and not res.get('note') else ''} · {res['attribution']}"
                           " · skill match uses the short job snippet Adzuna provides")
                job_cards(score_jobs(jobs, oi))
                csv_df = pd.DataFrame([{"Title": j["title"], "Company": j["company"], "Location": j["location"], "Salary": j["salary"],
                                        "Posted": j["date"], "Apply link": j["url"], "Graph skills in posting": "; ".join(j["skills"])} for j in jobs])
                st.download_button("⬇ Download these openings (CSV)", csv_df.to_csv(index=False).encode("utf-8-sig"),
                                   file_name=f"adzuna_{occ['id']}_{country}_{where or 'all'}.csv", mime="text/csv")
            except live_jobs.LiveJobsError as e:
                st.warning(str(e))
            except Exception as e:
                st.warning(f"Adzuna could not be reached ({type(e).__name__}). Check your internet connection, or use the 2024 dataset tab.")
        else:
            st.caption("Choose a country and city, then press Search. Needs a free Adzuna key in .env (see README).")
    with t2:
        if st.button("Load remote openings", key="remGo") or ss.get("rem_key") == occ["id"]:
            ss.rem_key = occ["id"]
            try:
                with st.spinner("Loading remote openings…"):
                    res = live_jobs.remotive(occ["id"])
                jobs = [{**j, "skills": E.extract(j.pop("text"))} for j in [dict(x) for x in res["jobs"]]]
                st.caption(f"{len(jobs)} remote openings · {res['attribution']}")
                job_cards(score_jobs(jobs, oi))
                st.download_button("⬇ Download these openings (CSV)", pd.DataFrame([{"Title": j["title"], "Company": j["company"], "Location": j["location"],
                                   "Salary": j["salary"], "Posted": j["date"], "Apply link": j["url"]} for j in jobs]).to_csv(index=False).encode("utf-8-sig"),
                                   file_name=f"remotive_{occ['id']}.csv", mime="text/csv", key="remDl")
            except Exception as e:
                st.warning(f"Remotive could not be reached ({type(e).__name__}). Check your internet connection.")
    with t3:
        data = openings_data()["occupations"].get(occ["id"])
        if not data:
            st.write("No dataset openings for this occupation.")
        else:
            s = data["stats"]; exp_tot = sum(s["experience"].values()) or 1
            c = st.columns(4)
            c[0].metric("Median salary", f"${round(s['salary_median'] / 1000)}k" if s["salary_median"] else "—",
                        help=f"25th–75th percentile ${round((s['salary_p25'] or 0) / 1000)}k–${round((s['salary_p75'] or 0) / 1000)}k, from {s['salary_n']} postings with a salary")
            c[1].metric("Remote allowed", f"{s['remote_pct']}%"); c[2].metric("Postings", f"{s['postings']:,}")
            c[3].metric("Entry level", f"{round(100 * s['experience'].get('Entry level', 0) / exp_tot)}%")
            st.markdown("**Top hiring companies:** " + chips([f"{k} · {v}" for k, v in s["top_companies"]], "sky"), unsafe_allow_html=True)
            st.markdown("**Top locations:** " + chips([f"{k} · {v}" for k, v in s["top_locations"]]), unsafe_allow_html=True)
            opts = ["all", "remote"] + [k for k, _ in s["top_states"]]
            loc = st.selectbox("Location", opts, key="dsloc", format_func=lambda k: {"all": "All locations", "remote": "Remote only"}.get(k, k))
            jobs = [j for j in data["jobs"] if loc == "all" or (loc == "remote" and j["remote"]) or j["state"] == loc]
            job_cards(score_jobs(jobs, oi))
            st.caption(openings_data().get("note", ""))

def careers():
    st.title("Careers ranked for you")
    if not U():
        st.warning("Your profile has no skills from the graph yet, so every occupation scores the same. Add skills on **Profile**.")
    dom = st.radio("Domain", ["All"] + list(DOMAIN_COL), horizontal=True)
    R = ranked(); rows = [r for r in R if dom == "All" or r["domain"] == dom]
    df = pd.DataFrame([{"Rank": R.index(r) + 1, "Occupation": r["name"], "Domain": r["domain"], "HGT": r["hgt"], "Node2Vec": r["node2vec"],
                        "Coverage": r["coverage"]} for r in rows])
    st.dataframe(df, hide_index=True, use_container_width=True, height=300, column_config={
        "HGT": st.column_config.NumberColumn(format="%.3f"), "Node2Vec": st.column_config.NumberColumn(format="%.3f"),
        "Coverage": st.column_config.ProgressColumn("Weighted skill coverage", min_value=0, max_value=1, format="percent")})
    pick = st.selectbox("Open an occupation", [r["id"] for r in rows], format_func=lambda i: OCC_BY_ID[i][1]["name"])
    if not pick: return
    oi, o = OCC_BY_ID[pick]; r = next(x for x in R if x["id"] == pick)
    st.divider()
    pic, txt = st.columns([1, 2])
    pic.image(occ_img(o), use_container_width=True)
    with txt:
        st.markdown(f"## {o['name']} &nbsp; {dchip(o['domain'])}", unsafe_allow_html=True)
        st.caption(f"O*NET {o['id']} · {o['onet_title']}")
        st.write(o["description"])
    c = st.columns(4)
    c[0].metric(f"Rank by {MODELS[ss.model]}", f"#{R.index(r) + 1} of 27"); c[1].metric("HGT score", f"{r['hgt']:.3f}")
    c[2].metric("Node2Vec score", f"{r['node2vec']:.3f}"); c[3].metric("Weighted coverage", f"{round(r['coverage'] * 100)}%")
    st.subheader("🧠 Skill gap")
    st.caption("Weight = confidence × (0.3 + 0.7 × demand ÷ highest demand). Demand = freshness-weighted share of postings mentioning the skill.")
    st.dataframe(gap_table(oi), hide_index=True, use_container_width=True, column_config={
        "Weight": st.column_config.ProgressColumn(min_value=0, max_value=1, format="%.2f"), "Confidence": st.column_config.NumberColumn(format="%.2f")})
    ex = E.explain(ss.skills, pick, n=8)
    st.subheader("Why this recommendation")
    st.markdown("**Matched:** " + chips([f"{m['skill']} · {m['weight']:.2f}" for m in ex["matched"][:8]], "have"), unsafe_allow_html=True)
    st.markdown("**Top missing:** " + chips([f"{m['skill']} · {m['weight']:.2f}" for m in ex["missing"][:6]], "gap"), unsafe_allow_html=True)
    for p in ex["paths"]:
        st.markdown(f'<div class="kgpath {"rel" if "RELATED_TO" in p else ""}">{p}</div>', unsafe_allow_html=True)
    st.subheader("📚 Courses for your missing skills")
    if ex["courses"]:
        for k in ex["courses"]:
            st.markdown(f"- [{k['course']}]({k['url']}) — teaches **{k['teaches']}**")
    else:
        st.write("No course in the catalogue teaches these skills.")
    if o.get("tasks"):
        with st.expander("What the work involves (O*NET tasks)"):
            for t in o["tasks"]: st.markdown(f"- {t}")
    openings_section(o, oi)

def roadmap():
    st.title("🗺️ Learning roadmap")
    R = ranked(); order = [r["id"] for r in R]
    pick = st.selectbox("Route to", order, format_func=lambda i: OCC_BY_ID[i][1]["name"])
    oi, o = OCC_BY_ID[pick]; have = set(U())
    st.image(occ_img(o), use_container_width=True)
    req = sorted(E.req[oi].items(), key=lambda kv: -kv[1]["w"]); miss = [(s, e) for s, e in req if s not in have and e["w"] >= 0.15]
    quick = [(s, e) for s, e in miss if any(x in have for x, _ in E.rel[s])][:5]
    core = [(s, e) for s, e in miss if (s, e) not in quick][:6]
    crs = list(dict.fromkeys(DB["prepares"].get(pick, []) + [c for s, _ in quick + core for c in E.teachers[s][:1]]))[:6]
    prog = ss.progress.setdefault(pick, {})
    cov = next(x for x in R if x["id"] == pick)["coverage"]
    st.markdown(f"You cover **{round(cov * 100)}%** of {o['name']}'s weighted skills. The plan starts with missing skills the graph links to ones you already have.")
    stages = [("1 · Quick wins", "Missing skills RELATED_TO a skill you have.", [(f"s:{s}", SK[s], f"weight {e['w']:.2f} · via {SK[next(x for x, _ in E.rel[s] if x in have)]}") for s, e in quick]),
              ("2 · Core gaps", "The most heavily weighted skills you are still missing.", [(f"s:{s}", SK[s], f"weight {e['w']:.2f} · in {round(100 * e['d'])}% of postings") for s, e in core]),
              ("3 · Courses", "PREPARES_FOR this occupation, or TEACHES your missing skills.", [(f"c:{c}", DB["courses"][c]["name"], DB["courses"][c]["org"]) for c in crs]),
              ("4 · Show the work", "Portfolio projects from the occupation's O*NET tasks.", [(f"p:{i}", t, "") for i, t in enumerate(o.get("tasks", [])[:3])])]
    total = sum(len(s[2]) for s in stages); done = sum(1 for k in prog if prog[k])
    st.progress(done / total if total else 0, text=f"{done} of {total} steps done")
    for title, sub, items in stages:
        with st.container(border=True):
            st.markdown(f"#### {title}"); st.caption(sub)
            if not items: st.write("Nothing here — you already cover this stage.")
            for key, label, meta in items:
                v = st.checkbox(f"{label}" + (f"  ·  {meta}" if meta else ""), value=prog.get(key, False), key=f"{pick}-{key}")
                if v != prog.get(key, False):
                    prog[key] = v
                    if v and key.startswith("s:") and SK[int(key[2:])] not in ss.skills:
                        ss.skills = ss.skills + [SK[int(key[2:])]]; st.toast(f"{SK[int(key[2:])]} added to your skills"); st.rerun()

@st.cache_data
def ring_layout(dom, min_w):
    occ = sorted([(i, o) for i, o in enumerate(OCC) if dom == "All" or o["domain"] == dom], key=lambda t: (t[1]["domain"], t[1]["name"]))
    G = nx.Graph(); pos = {}
    for k, (i, o) in enumerate(occ):
        a = 2 * math.pi * k / len(occ) - math.pi / 2; G.add_node(("o", i)); pos[("o", i)] = np.array([math.cos(a), math.sin(a)])
    for i, _ in occ:
        for s, e in E.req[i].items():
            if e["w"] >= min_w: G.add_edge(("o", i), ("s", s), weight=e["w"])
    init = {n: pos.get(n, np.random.default_rng(1).normal(0, .2, 2)) for n in G}
    p = nx.spring_layout(G, pos=init, fixed=[n for n in G if n[0] == "o"], k=0.3, iterations=200, seed=3, weight="weight")
    rmax = max((float(np.hypot(*p[n])) for n in G if n[0] == "s"), default=1.0)
    for n in G:                                   # keep every skill inside the ring of occupations
        if n[0] == "s" and rmax > 0.8: p[n] = p[n] * (0.8 / rmax)
    return [(n, tuple(p[n])) for n in G], [(u, v, d["weight"]) for u, v, d in G.edges(data=True)]

def graph():
    st.title("🌐 Knowledge graph")
    c1, c2 = st.columns([1, 1]); dom = c1.selectbox("Domain", ["All"] + list(DOMAIN_COL)); min_w = c2.slider("Minimum edge weight", 0.2, 0.8, 0.4, 0.05)
    nodes, edges = ring_layout(dom, min_w); P = dict(nodes); have = set(U())
    fig = go.Figure()
    for d, col in DOMAIN_COL.items():
        xs, ys = [], []
        for u, v, w in edges:
            o = u if u[0] == "o" else v
            if OCC[o[1]]["domain"] == d: xs += [P[u][0], P[v][0], None]; ys += [P[u][1], P[v][1], None]
        if xs: fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", line=dict(color=col, width=0.7), opacity=.35, hoverinfo="skip", showlegend=False))
    deg = {}
    for u, v, _ in edges: deg[u] = deg.get(u, 0) + 1; deg[v] = deg.get(v, 0) + 1
    sk = [n for n, _ in nodes if n[0] == "s"]
    fig.add_trace(go.Scatter(x=[P[n][0] for n in sk], y=[P[n][1] for n in sk], mode="markers+text", name="Skill (teal = on your profile)",
                             text=[SK[n[1]] if deg.get(n, 0) >= 5 else "" for n in sk], textposition="top center", textfont=dict(size=10),
                             marker=dict(size=[6 + 1.2 * deg.get(n, 0) for n in sk], color=["#1f7a74" if n[1] in have else "#f2a541" for n in sk], line=dict(color="white", width=1)),
                             hovertext=[f"{SK[n[1]]} · required by {deg.get(n, 0)} occupations" + (" · on your profile" if n[1] in have else "") for n in sk], hoverinfo="text"))
    for d, col in DOMAIN_COL.items():
        on = [n for n, _ in nodes if n[0] == "o" and OCC[n[1]]["domain"] == d]
        if on: fig.add_trace(go.Scatter(x=[P[n][0] for n in on], y=[P[n][1] for n in on], mode="markers+text", name=d,
                                        text=[OCC[n[1]]["name"] for n in on], textposition="middle right", textfont=dict(color=col, size=11),
                                        marker=dict(size=14, color=col, line=dict(color="white", width=1.5)), hoverinfo="text"))
    fig.update_layout(height=720, margin=dict(l=10, r=10, t=10, b=10), xaxis=dict(visible=False), yaxis=dict(visible=False, scaleanchor="x"),
                      plot_bgcolor="#f8faf9", legend=dict(orientation="h", y=-0.02))
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Occupations on the ring, coloured by domain. Orange = skills (teal = on your profile). Shared skills sit in the centre.")
    st.subheader("One course in focus")
    course_names = sorted({DB["courses"][c]["name"] for c in range(len(DB["courses"])) if len(DB["courses"][c]["teaches"]) >= 2})
    default = course_names.index("Machine Learning with TensorFlow on Google Cloud Platform") if "Machine Learning with TensorFlow on Google Cloud Platform" in course_names else 0
    cname = st.selectbox("Course", course_names, index=default)
    ci = next(i for i, c in enumerate(DB["courses"]) if c["name"] == cname); skills = DB["courses"][ci]["teaches"]
    links = [(s, i, E.req[i][s]["w"]) for s in skills for i in range(len(OCC)) if s in E.req[i] and E.req[i][s]["w"] >= 0.3]
    occs = sorted({i for _, i, _ in links}, key=lambda i: -sum(w for s, j, w in links if j == i))[:10]
    f2 = go.Figure(); sy = {s: 1 - (k + .5) / len(skills) for k, s in enumerate(skills)}; oy = {i: 1 - (k + .5) / max(len(occs), 1) for k, i in enumerate(occs)}
    for s in skills: f2.add_trace(go.Scatter(x=[0, 1], y=[.5, sy[s]], mode="lines", line=dict(color="#D64545", width=2), hoverinfo="skip", showlegend=False))
    for s, i, w in links:
        if i in oy: f2.add_trace(go.Scatter(x=[1, 2], y=[sy[s], oy[i]], mode="lines", line=dict(color=DOMAIN_COL[OCC[i]["domain"]], width=1 + 4 * w), opacity=.5, hoverinfo="skip", showlegend=False))
    f2.add_trace(go.Scatter(x=[0], y=[.5], mode="markers+text", text=["Course"], textposition="top center", marker=dict(size=28, color="#D64545"), hovertext=[cname], hoverinfo="text", showlegend=False))
    f2.add_trace(go.Scatter(x=[1] * len(skills), y=[sy[s] for s in skills], mode="markers+text", text=[SK[s] for s in skills], textposition="top center", marker=dict(size=18, color="#F28E2B"), showlegend=False))
    f2.add_trace(go.Scatter(x=[2] * len(occs), y=[oy[i] for i in occs], mode="markers+text", text=[OCC[i]["name"] for i in occs], textposition="middle right",
                            marker=dict(size=12, color=[DOMAIN_COL[OCC[i]["domain"]] for i in occs]), showlegend=False))
    f2.update_layout(height=430, margin=dict(l=10, r=10, t=30, b=10), xaxis=dict(visible=False, range=[-.3, 3.2]), yaxis=dict(visible=False, range=[-.05, 1.05]),
                     plot_bgcolor="white", annotations=[dict(x=0, y=1.06, text="COURSE", showarrow=False), dict(x=1, y=1.06, text="TEACHES → skills", showarrow=False),
                                                        dict(x=2.3, y=1.06, text="← REQUIRED BY occupations", showarrow=False)])
    st.plotly_chart(f2, use_container_width=True)

def results():
    st.title("📊 Evaluation results")
    st.caption("Mean ± standard deviation over 3 random seeds. Test edges and held-out postings were never seen by Node2Vec or HGT.")
    S = dict(DB["summary"]); s = DB["stats"]; n = DB["n_profiles"]
    xp = ROOT / "data" / "experiments.json"; X = json.loads(xp.read_text(encoding="utf-8"))["summary"] if xp.exists() else None
    if X: S["LP/REQUIRES/Popularity"] = X["exp2_link_prediction"]["Popularity"]      # stronger popularity baseline (REQUIRES degree)
    fmt = lambda k, m: f"{S[k][m]['mean']:.3f} ± {S[k][m]['sd']:.3f}"
    st.subheader("Knowledge graph")
    c = st.columns(6)
    for col, (lab, val) in zip(c, [("Nodes", f"{s['nodes']['total']:,}"), ("Edges", f"{s['edges']['total']:,}"), ("Avg. degree", s["avg_degree"]),
                                   ("Largest component", f"{s['lcc_pct']}%"), ("Schema violations", s["schema_violations"]), ("Duplicates/dangling", s["duplicate_edges"] + s["dangling_edges"])]):
        col.metric(lab, val)
    st.subheader("Link prediction (held-out edges)")
    st.caption(f"{n['n_test']['REQUIRES']} occupation→skill and {n['n_test']['TEACHES']} course→skill test edges; filtered ranking against all 475 skills.")
    lp = pd.DataFrame([{"Relation": rel, "Model": m, "AUC": fmt(f"LP/{k}/{m}", "AUC"), "Hits@10": fmt(f"LP/{k}/{m}", "Hits@10"), "MRR": fmt(f"LP/{k}/{m}", "MRR")}
                       for rel, k in [("Occupation → skill", "REQUIRES"), ("Course → skill", "TEACHES")] for m in ["Popularity", "Node2Vec", "HGT"]])
    if X:
        cf = pd.DataFrame([{"Relation": "Occupation → skill", "Model": m, **{k: f"{X['exp2_link_prediction'][m][k]['mean']:.3f} ± {X['exp2_link_prediction'][m][k]['sd']:.3f}" for k in ["AUC", "Hits@10", "MRR"]}}
                           for m in ["Item-based CF", "User-based CF"]])
        lp = pd.concat([lp.iloc[:1], cf, lp.iloc[1:]], ignore_index=True)
    st.dataframe(lp, hide_index=True, use_container_width=True)
    fig = go.Figure()
    for m, col in [("Popularity", "#c9d5d0"), ("Node2Vec", "#e2b64b"), ("HGT", "#7b61b8")]:
        fig.add_trace(go.Bar(name=m, marker_color=col, x=["AUC occ→skill", "MRR occ→skill", "AUC course→skill", "MRR course→skill"],
                             y=[S[f"LP/REQUIRES/{m}"]["AUC"]["mean"], S[f"LP/REQUIRES/{m}"]["MRR"]["mean"], S[f"LP/TEACHES/{m}"]["AUC"]["mean"], S[f"LP/TEACHES/{m}"]["MRR"]["mean"]]))
    fig.update_layout(barmode="group", height=330, margin=dict(l=10, r=10, t=10, b=10), yaxis_range=[0, 1])
    st.plotly_chart(fig, use_container_width=True)
    st.info("HGT clearly improves on Node2Vec. For occupation→skill links, popularity and collaborative filtering beat HGT on Hits@10 and MRR, because common skills are easy guesses; HGT gives the best AUC.")
    st.subheader("Occupation recommendation")
    rec = pd.DataFrame([{"Profiles": lab, "Model": m, **{k: fmt(f"REC/{p}/{m}", k) for k in ["Hits@1", "MRR", "Recall@10", "NDCG@10"]}}
                        for lab, p in [(f"Simulated ({n['n_profiles/simulated']})", "simulated"), (f"Real held-out postings ({n['n_profiles/heldout_postings']})", "heldout_postings")]
                        for m in ["Random", "Skill overlap", "Node2Vec", "HGT"]])
    st.dataframe(rec, hide_index=True, use_container_width=True)
    fig = go.Figure()
    for m, col in [("Random", "#e3e8e6"), ("Skill overlap", "#9db3d6"), ("Node2Vec", "#e2b64b"), ("HGT", "#7b61b8")]:
        fig.add_trace(go.Bar(name=m, marker_color=col, x=["MRR · simulated", "Recall@10 · simulated", "MRR · real postings", "Recall@10 · real postings"],
                             y=[S[f"REC/simulated/{m}"]["MRR"]["mean"], S[f"REC/simulated/{m}"]["Recall@10"]["mean"], S[f"REC/heldout_postings/{m}"]["MRR"]["mean"], S[f"REC/heldout_postings/{m}"]["Recall@10"]["mean"]]))
    fig.update_layout(barmode="group", height=330, margin=dict(l=10, r=10, t=10, b=10), yaxis_range=[0, 1])
    st.plotly_chart(fig, use_container_width=True)
    st.info("HGT is best on simulated profiles, which come from the same graph it trains on. On the same real held-out postings Node2Vec is slightly ahead (0.558 vs 0.535); "
            "both clearly beat skill overlap (0.419) and random ranking (0.140). The held-out postings are the more realistic estimate.")
    if X:
        f2 = lambda d: f"{d['mean']:.3f} ± {d['sd']:.3f}"
        st.subheader("Compared with standard recommenders")
        st.caption("Same profiles and seeds. Collaborative filtering cannot rank occupations for a new learner (no interaction history), so it appears only in link prediction.")
        rows = [{"Method": m, "MRR (simulated)": f2(d["simulated"]["MRR"]), "MRR (real)": f2(d["heldout_postings"]["MRR"]),
                 "Recall@10 (real)": f2(d["heldout_postings"]["Recall@10"]), "NDCG@10 (real)": f2(d["heldout_postings"]["NDCG@10"])} for m, d in X["exp2_recommendation"].items() if m != "TF-IDF content-based"]
        for m in ["Node2Vec", "HGT"]:
            rows.append({"Method": f"{m} (ours)", "MRR (simulated)": fmt(f"REC/simulated/{m}", "MRR"), "MRR (real)": fmt(f"REC/heldout_postings/{m}", "MRR"),
                         "Recall@10 (real)": fmt(f"REC/heldout_postings/{m}", "Recall@10"), "NDCG@10 (real)": fmt(f"REC/heldout_postings/{m}", "NDCG@10")})
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
        st.info("The real held-out postings are the more realistic estimate; simulated profiles are sampled from the graph's own occupation-skill edges.")
        st.subheader("Edge-weighting ablation (real held-out postings)")
        lab = {"uniform": "Uniform (no weighting)", "confidence_only": "Confidence only", "demand_only": "Demand only", "no_freshness": "No freshness damping", "full": "Full (proposed)"}
        models = [m for m in ["Node2Vec", "HGT", "Weighted skill overlap"] if m in X["exp1"]["full"]]
        st.dataframe(pd.DataFrame([{"Weighting": lab[v], **{f"{m} {k}": f2(X["exp1"][v][m]["heldout_postings"][k]) for m in models for k in ["MRR", "Recall@10"]}}
                                   for v in lab if v in X["exp1"]]), hide_index=True, use_container_width=True)
        st.info("The proposed weighting improves Node2Vec by about 15% and skill overlap by about 54% over an unweighted graph; confidence and demand each help. "
                "Removing posting-freshness damping makes no measurable difference, because the postings span only about two weeks.")
    st.subheader("Course retrieval for a missing skill")
    cr = pd.DataFrame([{"Model": m, **{k: fmt(f"COURSE/{m}", k) for k in ["MRR", "Recall@10", "NDCG@10"]}} for m in ["Popularity", "Node2Vec", "HGT"]])
    st.dataframe(cr, hide_index=True, use_container_width=True)
    st.caption(f"Only {n['n_test']['course_queries']} test queries, so treat these as indicative.")

pg = st.navigation([st.Page(home, title="Home", icon="🏠", default=True), st.Page(profile, title="Profile", icon="👤"),
                    st.Page(careers, title="Careers", icon="🎯"), st.Page(roadmap, title="Roadmap", icon="🗺️"),
                    st.Page(graph, title="Knowledge graph", icon="🌐"), st.Page(results, title="Results", icon="📊")])
pg.run()
