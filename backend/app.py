"""
app.py — serves SkillGraph and a JSON API.
    pip install -r requirements.txt
    python backend/app.py            → http://127.0.0.1:5000   (VS Code: F5 ▸ "▶ Run SkillGraph")
API: GET /api/live-openings?occupation=&source=adzuna|remotive&country=in&where=Chennai · GET /api/openings?occupation=&location= · GET /api/health · GET /api/occupations · POST /api/recommend {skills, model} · POST /api/explain {skills, occupation}
     GET /api/metrics · POST /api/resume (multipart "file")
"""
import io, json, mimetypes, os
from pathlib import Path
from flask import Flask, jsonify, request, send_from_directory, abort
from kg_engine import Engine
from live_jobs import adzuna, remotive, LiveJobsError

def _load_env(path):                     # tiny .env reader, so no extra package is needed
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if "=" in line and not line.strip().startswith("#"):
                k, v = line.split("=", 1); os.environ.setdefault(k.strip(), v.strip().strip('"'))

mimetypes.add_type("application/javascript", ".js"); mimetypes.add_type("text/css", ".css"); mimetypes.add_type("application/json", ".json")
ROOT = Path(__file__).resolve().parent.parent
_load_env(ROOT / ".env")
if not (ROOT / "data" / "kg_web.json").exists():
    raise SystemExit("data/kg_web.json is missing. Run the pipeline (see README) or restore the data folder.")
app = Flask(__name__, static_folder=None); app.config["MAX_CONTENT_LENGTH"] = 8 * 1024 * 1024
eng = Engine()
body = lambda: request.get_json(silent=True) or {}

@app.get("/api/health")
def health(): return {"ok": True, "occupations": len(eng.db["occupations"]), "skills": len(eng.db["skills"]), "courses": len(eng.db["courses"])}

@app.get("/api/occupations")
def occupations(): return jsonify([{k: o[k] for k in ("id", "name", "domain", "onet_title", "postings")} | {"n_skills": len(o["requires"])} for o in eng.db["occupations"]])

@app.post("/api/recommend")
def recommend():
    b = body(); model = b.get("model", "hgt")
    if model not in ("hgt", "node2vec", "overlap"): return {"error": "model must be hgt, node2vec or overlap"}, 400
    return jsonify(eng.rank(b.get("skills", []), model, int(request.args.get("top", 10))))

@app.post("/api/explain")
def explain():
    b = body()
    try: return jsonify(eng.explain(b.get("skills", []), b.get("occupation", "")))
    except StopIteration: abort(404)

OPEN = json.loads((ROOT / "data" / "openings.json").read_text(encoding="utf-8")) if (ROOT / "data" / "openings.json").exists() else {"occupations": {}}

@app.get("/api/openings")
def openings():
    occ = request.args.get("occupation", ""); loc = request.args.get("location", "all")
    d = OPEN["occupations"].get(occ)
    if not d: abort(404)
    jobs = [j for j in d["jobs"] if loc == "all" or (loc == "remote" and j["remote"]) or j["state"] == loc]
    return jsonify(dict(stats=d["stats"], jobs=jobs, note=OPEN.get("note", "")))

@app.get("/api/live-locations")
def live_locations():
    """Countries and cities for the live-openings dropdowns."""
    from live_jobs import LOCATIONS
    return jsonify(LOCATIONS)


@app.get("/api/live-openings")
def live_openings():
    occ = request.args.get("occupation", ""); src = request.args.get("source", "adzuna")
    try:
        res = remotive(occ) if src == "remotive" else adzuna(occ, request.args.get("country", "in"), request.args.get("where", ""))
    except LiveJobsError as e:
        return {"error": str(e)}, 503
    except Exception as e:                                  # network down, API changed, rate limited …
        return {"error": f"The {src} service could not be reached ({type(e).__name__}). Try again later or use the 2024 dataset view."}, 502
    res = {**res, "jobs": [dict(j) for j in res["jobs"]]}
    for j in res["jobs"]:
        j["skills"] = eng.extract(j.pop("text", ""))       # same skill lexicon as the knowledge graph
    return jsonify(res)

@app.get("/api/metrics")
def metrics():
    exp = ROOT / "data" / "experiments.json"
    return jsonify(dict(stats=eng.db["stats"], summary=eng.db["summary"], n_profiles=eng.db["n_profiles"],
                        experiments=json.loads(exp.read_text(encoding="utf-8"))["summary"] if exp.exists() else None))

@app.post("/api/resume")
def resume():
    f = request.files.get("file")
    if not f: return {"error": "Attach a file in the 'file' field."}, 400
    name, data = f.filename.lower(), f.read()
    try:
        if name.endswith(".pdf"):
            from pypdf import PdfReader; text = "\n".join(p.extract_text() or "" for p in PdfReader(io.BytesIO(data)).pages)
        elif name.endswith(".docx"):
            import docx; text = "\n".join(p.text for p in docx.Document(io.BytesIO(data)).paragraphs)
        else: text = data.decode("utf-8", "ignore")
    except ImportError: return {"error": "Install pypdf and python-docx to read PDF/DOCX on the server."}, 500
    return jsonify({"text": text, "skills": eng.extract(text)})

@app.get("/")
def index(): return send_from_directory(ROOT, "index.html")

@app.get("/<path:path>")
def static_files(path):
    if path.startswith(("backend", "pipeline", ".vscode", ".venv", "raw")): abort(404)
    return send_from_directory(ROOT, path)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"\n  SkillGraph is running →  http://127.0.0.1:{port}\n  Press Ctrl+C to stop.\n")
    app.run(host="127.0.0.1", port=port, debug=True, use_reloader=False)
