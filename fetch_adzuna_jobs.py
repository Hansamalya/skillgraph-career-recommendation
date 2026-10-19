"""
fetch_adzuna_jobs.py — download CURRENT job openings from Adzuna for all 27 occupations into a CSV.

Usage (from the project folder, with your free key in .env):
    python fetch_adzuna_jobs.py --country in --where Chennai
    python fetch_adzuna_jobs.py --country in --where Bengaluru --per 50
    python fetch_adzuna_jobs.py --country us --where "New York" --occ 15-2051.00 15-2051.01

Output: data/adzuna_openings_<country>_<where>_<date>.csv  (opens in Excel; UTF-8 with BOM so ₹ shows correctly)
Columns: occupation, domain, title, company, location, salary_min, salary_max, currency, contract_time,
         posted_date, apply_link, graph_skills_in_posting
Key setup: register free at https://developer.adzuna.com, then put ADZUNA_APP_ID and ADZUNA_APP_KEY in .env.
One run = one API call per occupation (27 calls), well inside Adzuna's free limit (about 1,000 calls a month).
Results come from Adzuna; show "Jobs by Adzuna" wherever you display them.
"""
import argparse, csv, datetime, os, re, sys, time
from pathlib import Path
import requests

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "backend"))
from kg_engine import Engine                  # noqa: E402  (same skill lexicon as the knowledge graph)
from live_jobs import QUERY, CURRENCY, ALT    # noqa: E402

def load_env(path=ROOT / ".env"):
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if "=" in line and not line.strip().startswith("#"):
                k, v = line.split("=", 1); os.environ.setdefault(k.strip(), v.strip().strip('"'))

def clean(t):
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", t or "")).strip()

def main(argv=None):
    ap = argparse.ArgumentParser(description="Download current Adzuna openings for the 27 occupations into a CSV.")
    ap.add_argument("--country", default="in", help="in, us, gb, sg, au, ca … (default: in)")
    ap.add_argument("--where", default="Chennai", help='city or region, e.g. Chennai, Bengaluru, "New York" (default: Chennai)')
    ap.add_argument("--per", type=int, default=20, help="openings per occupation, max 50 (default: 20)")
    ap.add_argument("--days", type=int, default=30, help="only postings from the last N days (default: 30)")
    ap.add_argument("--occ", nargs="*", help="only these O*NET codes (default: all 27)")
    a = ap.parse_args(argv)
    load_env()
    app_id, app_key = os.environ.get("ADZUNA_APP_ID"), os.environ.get("ADZUNA_APP_KEY")
    if not app_id or not app_key:
        print("No Adzuna key found.\n  1. Register free at https://developer.adzuna.com\n  2. Copy .env.example to .env\n"
              "  3. Fill in ADZUNA_APP_ID and ADZUNA_APP_KEY, then run this again."); return 1
    eng = Engine(); occs = {o["id"]: o for o in eng.db["occupations"]}
    codes = a.occ or list(occs)
    cur = CURRENCY.get(a.country, "")
    rows, total = [], 0
    print(f"Searching Adzuna ({a.country.upper()}, {a.where or 'anywhere'}, last {a.days} days) for {len(codes)} occupations …")
    for code in codes:
        if code not in occs or code not in QUERY:
            print(f"  skip {code}: not one of the 27 occupations"); continue
        # title in the city → related title in the city → title country-wide (stops at the first search with results)
        attempts = [(QUERY[code], a.where, "city")] + [(alt, a.where, "city, related title") for alt in ALT.get(code, [])[:1]]
        if a.where: attempts.append((QUERY[code], "", "country-wide"))
        res, scope, stop = [], "", False
        for what, where, scope in attempts:
            r = requests.get(f"https://api.adzuna.com/v1/api/jobs/{a.country}/search/1", timeout=20,
                             params={"app_id": app_id, "app_key": app_key, "what": what, "where": where,
                                     "results_per_page": min(a.per, 50), "max_days_old": a.days, "sort_by": "date",
                                     "content-type": "application/json"})
            if r.status_code in (401, 403):
                print("Adzuna rejected the key. Check ADZUNA_APP_ID and ADZUNA_APP_KEY in .env."); return 1
            if r.status_code == 429:
                print("Adzuna rate limit reached; saving what was collected so far."); stop = True; break
            r.raise_for_status()
            res = r.json().get("results", [])
            if res: break
            time.sleep(1)
        if stop: break
        for j in res:
            title, desc = clean(j.get("title")), clean(j.get("description"))
            rows.append({"occupation": occs[code]["name"], "domain": occs[code]["domain"], "title": title,
                         "company": (j.get("company") or {}).get("display_name", ""),
                         "location": (j.get("location") or {}).get("display_name", ""),
                         "salary_min": int(j["salary_min"]) if j.get("salary_min") else "",
                         "salary_max": int(j["salary_max"]) if j.get("salary_max") else "", "currency": cur,
                         "contract_time": j.get("contract_time", "") or "", "posted_date": (j.get("created") or "")[:10],
                         "apply_link": j.get("redirect_url", ""), "search_scope": scope,
                         "graph_skills_in_posting": "; ".join(eng.extract(title + " " + desc))})
        total += len(res); print(f"  {occs[code]['name']:<40} {len(res):>3} openings" + (f"  ({scope})" if res and scope != "city" else ""))
        time.sleep(1)                                   # be polite to the free API
    if not rows:
        print("No openings found. Try another city (or --where \"\" for the whole country) or more --days."); return 0
    slug = re.sub(r"[^a-z0-9]+", "-", (a.where or "all").lower()).strip("-")
    out = ROOT / "data" / f"adzuna_openings_{a.country}_{slug}_{datetime.date.today()}.csv"
    out.parent.mkdir(exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print(f"\nSaved {total} openings to {out}\nJobs by Adzuna.")
    return 0

if __name__ == "__main__":
    sys.exit(main())
