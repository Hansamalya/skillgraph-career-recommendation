# SkillGraph — Knowledge Graph Embedding on Skill and Career Recommendation

Hansamalya B (25MDT1050) · Moniisha S (25MDT1112) · Guide: Dr. Manivannan A
M.Sc. Data Science, Department of Mathematics, School of Advanced Sciences, VIT Chennai

A website for the project: a heterogeneous knowledge graph of O*NET occupations, skills and Coursera courses, weighted with
LinkedIn job-posting demand, embedded with Node2Vec and refined with a Heterogeneous Graph Transformer (HGT).

## Run it in VS Code
1. **File ▸ Open Folder…** → this folder. Accept the recommended extensions.
2. **Terminal ▸ Run Task… ▸ 1. Setup: create .venv and install website requirements** (Flask, pypdf, python-docx — small).
   Select the `.venv` interpreter when VS Code asks.
3. **Run and Debug (Ctrl+Shift+D) ▸ ▶ Run SkillGraph ▸ F5.** The browser opens http://127.0.0.1:5000.
4. Click **Continue as guest**, then drop a résumé from `samples/` on the dashboard.

Other options: double-click `run.bat` (Windows) / `./run.sh` (macOS, Linux), or right-click `index.html` ▸ *Open with Live Server*.
Opening the HTML files directly from disk does not work: browsers block them from reading `data/kg_web.json`.

Tests: **Run and Debug ▸ ✔ Run tests** or `python backend/test_app.py` (8 tests: data, ranking, explanations, API, résumé upload).

## Pages
| Page | What it shows |
|---|---|
| Dashboard | Skills (autocomplete over 475 graph skills), résumé upload, model choice, top matches, graph analytics |
| Careers | 27 occupations ranked by **HGT**, **Node2Vec** or **skill overlap**; skill gap with weight, demand and confidence; matched / missing skills; RELATED_TO and TEACHES graph paths; courses |
| Roadmap | Quick wins (missing skills related to ones you have) → core gaps → courses → portfolio projects from O*NET tasks |
| Knowledge graph | Interactive D3 graph: occupations on a ring by domain, skills between them, optional courses |
| Results | All evaluation metrics (mean ± sd over 3 seeds) with honest notes |

## Logo, photos and banners
- `assets/logo.png`, `assets/logo.svg`, `assets/logo_wordmark.png`, `assets/favicon.png`: the SkillGraph logo (both apps).
- `assets/banners/`: built-in knowledge-graph artwork (hero + one banner per domain). Always available, even offline.
  Regenerate with `python assets/make_assets.py`.
- **Real photos:** run once with internet, `python fetch_images.py` (VS Code: **🖼 Download photos**). It saves the 27 occupation
  photos plus the hero and login photos to `assets/img/`, so they show offline during a demo. Photos: Unsplash (free licence).
- Each picture is tried in order: downloaded photo → online photo → domain banner, so nothing ever shows as broken.

## Streamlit version
The same app, built with Streamlit, is in `streamlit_app.py`. It uses the same engine, data and live-jobs module, so every ranking,
explanation and metric is identical to the Flask website.

**Run in VS Code:** Terminal ▸ Run Task ▸ **1b. Setup for Streamlit version**, then Run and Debug ▸ **▶ Run SkillGraph (Streamlit)** (F5).
The app opens at http://localhost:8501. Or double-click `run_streamlit.bat` (Windows) / `./run_streamlit.sh`, or run:
```bash
pip install -r requirements-streamlit.txt
streamlit run streamlit_app.py
```
Pages: Home · Profile (résumé upload, skills) · Careers (ranking, skill gap, explanations, courses, live/remote/2024 openings)
· Roadmap · Knowledge graph (ring view + one course in focus) · Results. The model (HGT / Node2Vec / Skill overlap) is chosen in the sidebar.
Live openings use the same `.env` Adzuna key. The profile is kept for the browser session.

## Live 2026 job openings (Adzuna + Remotive)
Each career page has three tabs under **Job openings**:

| Tab | Source | Needs |
|---|---|---|
| **Live 2026 · Adzuna** | Current openings by country and city (India, US, UK, Singapore, Australia, Canada) | A free Adzuna key |
| **Remote now · Remotive** | Current remote openings (listings delayed 24 h by Remotive) | Internet only, no key |
| **2024 dataset** | 1,464 sample postings + market snapshot from the Kaggle LinkedIn data | Nothing (works offline) |

**Set up the Adzuna key (2 minutes, free):**
1. Register at https://developer.adzuna.com and copy your *Application ID* and *Application Key*.
2. Copy `.env.example` to `.env` in the project folder and fill in both values.
3. Restart **▶ Run SkillGraph** (F5). The key stays on your machine; `.env` is ignored by git.

**Download a CSV of current openings (all 27 occupations):**
```bash
python fetch_adzuna_jobs.py --country in --where Chennai        # or: Bengaluru, Hyderabad, "" for all of India
```
It saves `data/adzuna_openings_in_chennai_<date>.csv` with occupation, title, company, location, salary, posted date,
apply link and the graph skills found in each posting. In VS Code: Run and Debug ▸ **⬇ Fetch Adzuna openings → CSV**.
The Live tab in both apps also has a **Download CSV** button for the openings on screen.

**Nothing shows in the Live tab?** Check, in order: (1) `.env` exists in the project folder (not only `.env.example`)
with both values filled in; (2) you restarted the app after creating it; (3) you are online; (4) try a bigger city
or leave the city empty. The tab shows the exact reason if the key is missing or rejected.

Notes: live tabs need the Flask server (not Live Server) and an internet connection. Results are cached for 6 hours
to respect the APIs' free limits (Adzuna: about 1,000 calls a month). Each posting's skills are matched with the same
lexicon as the knowledge graph, so "You have k of n skills" is comparable across all three tabs.
LinkedIn has no public jobs API and does not allow scraping, so live LinkedIn listings are not used.

## How the scores work (same as pipeline/src/kg_models.py)
- **HGT:** mean over your skills s of sigmoid(HGT DistMult score(occupation, s))
- **Node2Vec:** mean cosine similarity between the occupation and your skills
- **Skill overlap:** Σ weight of your skills ÷ Σ weight of all the occupation's skills
- **Edge weight** w = c · (0.3 + 0.7 · d / d_max), where d = freshness-weighted demand (posting-freshness damping, 14-day half-life) and
  c = 1 − (1 − e_O*NET)(1 − e_postings) is the confidence
- **Skill gap:** matched = REQUIRES(o) ∩ your skills; missing = REQUIRES(o) − your skills, ranked by w; courses = TEACHES edges

## Rebuilding the data (optional)
The website ships with `data/kg_web.json`. To regenerate it from the raw datasets:
```bash
pip install -r pipeline/requirements.txt          # PyTorch, PyTorch Geometric, gensim … (large)
python pipeline/src/kg_build.py raw               # phases 1–4  → pipeline/kg/
python pipeline/src/kg_models.py 0,1,2            # phases 5–9  → pipeline/results/metrics.json
python pipeline/src/kg_report.py                  # explanation example + figures
python pipeline/src/export_web.py raw             # → data/kg_web.json
```
`raw/` must contain `db_31_0_excel/`, `postings.csv` and `coursea_data.csv`. VS Code has a launch configuration for each step.

## Experiments: weighting ablation and standard recommenders
`pipeline/src/kg_experiments.py` tests the project's novelty on the same splits and profiles as the main evaluation:
- **Experiment 1:** edge-weighting ablation (uniform, confidence only, demand only, no freshness damping, full).
- **Experiment 2:** standard recommenders (random, popularity, Jaccard, TF-IDF content-based, weighted overlap; item- and user-based CF for link prediction).
```bash
pip install -r pipeline/requirements.txt          # includes PyTorch for the HGT rows
python pipeline/src/kg_experiments.py 0,1,2       # → pipeline/results/experiments.json and experiments.md
```
Without PyTorch the script still runs everything except the HGT rows. Results also appear on the Results page of both apps
(via `data/experiments.json`, written by `export_web.py`).

**Paper figures:** `python pipeline/src/kg_paper_figures.py` regenerates the IEEE paper's Fig. 2 (results) and Fig. 3
(case study: a learner with Python and SQL targeting Data Scientist) from the results files, so the figures always match
the numbers. Options: `--skills Python SQL --occupation 15-2051.00`.

**Terminology:** the 14-day half-life on postings is *posting-freshness damping*: the postings span about two weeks, so it
damps older postings within one snapshot rather than modelling long-term change (the ablation shows no measurable effect).

## API (backend/app.py)
`GET /api/health` · `GET /api/occupations` · `POST /api/recommend {"skills": [...], "model": "hgt"|"node2vec"|"overlap"}` ·
`POST /api/explain {"skills": [...], "occupation": "15-2051.00"}` · `GET /api/metrics` · `POST /api/resume` (multipart `file`)

## Limitations
Postings cover 15 days in 2024, mostly US; course–skill links come from course titles only; accounts are stored in the browser.
Simulated profiles are drawn from the graph itself, so the held-out-posting results on the Results page are the realistic estimate.
