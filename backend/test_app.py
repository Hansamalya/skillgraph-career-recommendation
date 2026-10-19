"""test_app.py — checks data, engine and API.   python backend/test_app.py"""
import io, os, sys, unittest
from pathlib import Path
HERE = Path(__file__).resolve().parent; ROOT = HERE.parent; sys.path.insert(0, str(HERE))
from kg_engine import Engine
from app import app

class Tests(unittest.TestCase):
    eng = Engine(); c = app.test_client()
    def test_data_shape(self):
        d = self.eng.db
        self.assertEqual(len(d["occupations"]), 27); self.assertEqual(len(d["scores"]["hgt"]), 27); self.assertEqual(len(d["scores"]["hgt"][0]), len(d["skills"]))
        self.assertEqual(d["stats"]["schema_violations"], 0)
        self.assertAlmostEqual(d["stats"]["avg_degree"], round(2 * d["stats"]["edges"]["total"] / d["stats"]["nodes"]["total"], 2))
    def test_site_files(self):
        for f in ("index.html", "login.html", "dashboard.html", "recommendation.html", "roadmap.html", "results.html", "graphs/knowledge_graph.html", "data/openings.json",
                  "js/app.js", "css/style.css", "js/vendor/d3.min.js", "js/vendor/chart.umd.min.js", "js/vendor/pdf.min.js", "js/vendor/mammoth.browser.min.js"):
            self.assertTrue((ROOT / f).exists(), f)
    def test_rank_models(self):
        sk = ["Python", "Machine Learning", "Deep Learning", "Statistics", "SQL"]
        for m in ("hgt", "node2vec", "overlap"):
            r = self.eng.rank(sk, m, 5); self.assertEqual(len(r), 5)
            self.assertIn("AI & Machine Learning", {x["domain"] for x in r}, m)
    def test_security_profile(self):
        top = [x["domain"] for x in self.eng.rank(["Network Security", "Penetration Testing", "Incident Response", "Vulnerability Management", "Splunk"], "hgt", 3)]
        self.assertIn("Cybersecurity", top)
    def test_explain(self):
        e = self.eng.explain(["Python", "SQL", "Statistics"], "15-2051.00")
        self.assertTrue(e["matched"]); self.assertTrue(e["missing"]); self.assertTrue(e["paths"])
    def test_extract(self):
        s = self.eng.extract((ROOT / "samples" / "resume_web_developer.txt").read_text(encoding="utf-8"))
        for k in ("JavaScript", "React", "Git"): self.assertIn(k, s)
    def test_api(self):
        self.assertTrue(self.c.get("/api/health").get_json()["ok"])
        self.assertEqual(len(self.c.post("/api/recommend?top=3", json={"skills": ["Figma", "Prototyping", "Wireframing"], "model": "hgt"}).get_json()), 3)
        self.assertEqual(self.c.post("/api/recommend", json={"skills": [], "model": "bad"}).status_code, 400)
        self.assertEqual(self.c.post("/api/explain", json={"skills": [], "occupation": "nope"}).status_code, 404)
        self.assertIn("summary", self.c.get("/api/metrics").get_json())
        for p in ("/", "/results.html", "/data/kg_web.json"): self.assertEqual(self.c.get(p).status_code, 200, p)
        self.assertEqual(self.c.get("/pipeline/src/kg_build.py").status_code, 404)
    def test_experiments(self):
        x = self.c.get("/api/metrics").get_json()["experiments"]
        self.assertIn("TF-IDF content-based", x["exp2_recommendation"])
        self.assertEqual(set(x["exp1"]), {"uniform", "confidence_only", "demand_only", "no_freshness", "full"})
        full = x["exp1"]["full"]["Node2Vec"]["heldout_postings"]["MRR"]["mean"]; uni = x["exp1"]["uniform"]["Node2Vec"]["heldout_postings"]["MRR"]["mean"]
        self.assertGreater(full, uni)                                      # the weighting helps (the paper's ablation claim)
    def test_openings(self):
        r = self.c.get("/api/openings?occupation=15-2051.00&location=remote").get_json()
        self.assertTrue(r["stats"]["postings"] > 0); self.assertTrue(all(j["remote"] for j in r["jobs"]))
        self.assertEqual(self.c.get("/api/openings?occupation=nope").status_code, 404)
    def test_live_openings_mocked(self):
        import live_jobs
        class R:
            def __init__(s, d): s.d, s.status_code = d, 200
            def json(s): return s.d
            def raise_for_status(s): pass
        az = {"results": [{"title": "Data Scientist", "company": {"display_name": "Acme"}, "location": {"display_name": "Chennai, Tamil Nadu"},
              "salary_min": 900000, "salary_max": 1500000, "created": "2026-10-01T10:00:00Z", "redirect_url": "https://example.com/1",
              "description": "Python, SQL and machine learning with Tableau dashboards"},
                   {"title": "Data Analyst", "company": {"display_name": "No Salary Listed"}, "location": {"display_name": "Chennai"},
                    "created": "2026-10-01T10:00:00Z", "redirect_url": "https://example.com/2", "description": "SQL and Excel"}]}
        rm = {"jobs": [{"title": "Senior Data Scientist", "company_name": "Remote Co", "candidate_required_location": "India",
              "publication_date": "2026-10-02T00:00:00", "url": "https://remotive.com/x", "description": "<p>Python and Statistics</p>"},
                       {"title": "Office Assistant", "company_name": "Other", "url": "https://remotive.com/y", "description": "Excel"}]}
        old_get, old_env = live_jobs.requests.get, dict(os.environ)
        live_jobs.requests.get = lambda url, **k: R(az if "adzuna" in url else rm); live_jobs.CACHE.clear()
        try:
            os.environ.pop("ADZUNA_APP_ID", None); os.environ.pop("ADZUNA_APP_KEY", None)
            self.assertEqual(self.c.get("/api/live-openings?occupation=15-2051.00").status_code, 503)        # no key → clear message
            os.environ["ADZUNA_APP_ID"], os.environ["ADZUNA_APP_KEY"] = "x", "y"
            r = self.c.get("/api/live-openings?occupation=15-2051.00&country=in&where=Chennai").get_json()
            self.assertEqual(r["jobs"][0]["company"], "Acme"); self.assertIn("₹", r["jobs"][0]["salary"]); self.assertIn("Python", r["jobs"][0]["skills"])
            self.assertEqual(r["jobs"][1]["salary"], "")
            repeated = self.c.get("/api/live-openings?occupation=15-2051.00&country=in&where=Chennai")
            self.assertEqual(repeated.status_code, 200)
            r = self.c.get("/api/live-openings?occupation=15-2051.00&source=remotive").get_json()
            self.assertEqual([j["title"] for j in r["jobs"]], ["Senior Data Scientist"])                    # unrelated job filtered out
        finally:
            live_jobs.requests.get = old_get; os.environ.clear(); os.environ.update(old_env)
    def test_fetch_adzuna_csv(self):
        sys.path.insert(0, str(ROOT)); import fetch_adzuna_jobs as fa, csv as _csv
        class R:
            status_code = 200
            def json(s): return {"results": [{"title": "Data Analyst", "company": {"display_name": "Acme"}, "location": {"display_name": "Chennai"},
                                              "salary_min": 600000, "salary_max": 900000, "created": "2026-10-05T09:00:00Z",
                                              "redirect_url": "https://www.adzuna.in/land/ad/1", "description": "SQL, Excel and Power BI"}]}
            def raise_for_status(s): pass
        old_get, old_env, old_sleep = fa.requests.get, dict(os.environ), fa.time.sleep
        fa.requests.get = lambda url, **k: R(); fa.time.sleep = lambda s: None
        try:
            os.environ.pop("ADZUNA_APP_ID", None); os.environ.pop("ADZUNA_APP_KEY", None); fa.load_env = lambda *a: None
            self.assertEqual(fa.main(["--occ", "15-2051.01"]), 1)                                   # no key → clear message, exit 1
            os.environ["ADZUNA_APP_ID"], os.environ["ADZUNA_APP_KEY"] = "x", "y"
            self.assertEqual(fa.main(["--country", "in", "--where", "Chennai", "--occ", "15-2051.01", "15-2051.00"]), 0)
            out = sorted((ROOT / "data").glob("adzuna_openings_in_chennai_*.csv"))[-1]
            rows = list(_csv.DictReader(open(out, encoding="utf-8-sig")))
            self.assertEqual(len(rows), 2); self.assertEqual(rows[0]["apply_link"], "https://www.adzuna.in/land/ad/1")
            self.assertEqual(rows[0]["currency"], "₹"); self.assertIn("SQL", rows[0]["graph_skills_in_posting"]); out.unlink()
        finally:
            fa.requests.get, fa.time.sleep = old_get, old_sleep; os.environ.clear(); os.environ.update(old_env)
    def test_live_fallback(self):
        import live_jobs
        class R:
            status_code = 200
            def __init__(s, d): s.d = d
            def json(s): return s.d
            def raise_for_status(s): pass
        calls = []
        def fake(url, params=None, **k):
            calls.append((params["what"], params["where"]))
            ok = params["what"] == "operations research analyst" and params["where"] == "Chennai"
            return R({"results": [{"title": "Operations Research Analyst", "company": {"display_name": "Acme"}, "location": {"display_name": "Chennai"},
                                   "created": "2026-10-10T00:00:00Z", "redirect_url": "https://x", "description": "optimization and Python"}]} if ok else {"results": []})
        old_get, old_env = live_jobs.requests.get, dict(os.environ)
        live_jobs.requests.get = fake; live_jobs.CACHE.clear(); os.environ["ADZUNA_APP_ID"], os.environ["ADZUNA_APP_KEY"] = "x", "y"
        try:
            r = live_jobs.adzuna("15-2031.00", "in", "Chennai")
            self.assertEqual(len(r["jobs"]), 1); self.assertIn("operations research", r["note"])
            self.assertEqual(calls, [("operations analyst", "Chennai"), ("operations research analyst", "Chennai")])   # stops at first hit
            live_jobs.requests.get = lambda url, **k: R({"results": []}); live_jobs.CACHE.clear()
            r = live_jobs.adzuna("15-2031.00", "in", "Chennai"); self.assertEqual(r["jobs"], []); self.assertIn("No current openings", r["note"])
        finally:
            live_jobs.requests.get = old_get; os.environ.clear(); os.environ.update(old_env)
    def test_live_locations(self):
        import json, re
        L = self.c.get("/api/live-locations").get_json()
        self.assertIn(["Chennai", "Chennai"], L["in"]["cities"]); self.assertIn(["Bangalore", "Bengaluru"], L["in"]["cities"])
        js = (ROOT / "js" / "app.js").read_text(encoding="utf-8")
        js_list = json.loads(re.search(r"const LIVE_LOCATIONS = (\{.*?\});\n", js).group(1))
        self.assertEqual(js_list, L)                                   # website dropdown = backend list
    def test_resume_upload(self):
        for ext in ("pdf", "docx", "txt"):
            f = ROOT / "samples" / f"resume_data_analyst.{ext}"
            r = self.c.post("/api/resume", data={"file": (io.BytesIO(f.read_bytes()), f.name)}, content_type="multipart/form-data")
            self.assertEqual(r.status_code, 200, ext); self.assertIn("SQL", r.get_json()["skills"], ext)

if __name__ == "__main__": unittest.main(verbosity=2)
