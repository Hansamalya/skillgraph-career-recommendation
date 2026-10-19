"""
live_jobs.py — current job openings from public job-search APIs (no scraping).

  Adzuna   : https://developer.adzuna.com  (free key; India, US, UK and more). Needs ADZUNA_APP_ID and ADZUNA_APP_KEY.
             Results must show "Jobs by Adzuna".
  Remotive : https://remotive.com/api/remote-jobs (no key; remote jobs only). Credit Remotive and link to each posting;
             results are delayed 24 h, and frequent polling is not allowed, so everything is cached for 6 hours.
LinkedIn has no public jobs API and does not permit scraping, so it is not used here.
"""
import os, re, time, html
import requests

CACHE, TTL = {}, 6 * 3600
QUERY = {  # what to search for each of the 27 occupations
 "15-1299.04": "penetration tester", "15-1299.06": "digital forensics", "15-1299.05": "security engineer", "15-1212.00": "security analyst",
 "15-1299.07": "blockchain developer", "15-1255.01": "game developer", "15-1255.00": "ux designer", "27-1024.00": "graphic designer",
 "15-2051.00": "data scientist", "15-1221.00": "machine learning engineer", "15-2041.00": "statistician", "15-2051.01": "data analyst",
 "15-2031.00": "operations analyst", "15-1243.00": "data architect", "15-1242.00": "data engineer", "15-1241.00": "cloud architect",
 "15-1299.08": "devops engineer", "15-1244.00": "system administrator", "11-3021.00": "it manager", "15-1254.00": "web developer",
 "15-1253.00": "qa engineer", "15-1251.00": "programmer", "15-1252.00": "software developer", "15-1211.00": "systems analyst",
 "13-1111.00": "business analyst", "13-1082.00": "project manager", "13-1161.01": "digital marketing"}
# Related titles tried when the main search finds nothing (Indian postings often use different titles).
# Each list stays within the same occupation, so a fallback never shows a different career's jobs.
ALT = {
 "15-1299.04": ["ethical hacker", "vapt"], "15-1299.06": ["forensic analyst", "cyber forensics"], "15-1299.05": ["cyber security engineer"],
 "15-1212.00": ["soc analyst", "cyber security analyst"], "15-1299.07": ["blockchain"], "15-1255.01": ["unity developer", "game designer"],
 "15-1255.00": ["ui ux designer", "product designer"], "27-1024.00": ["visual designer"], "15-2051.00": ["data science"],
 "15-1221.00": ["ai engineer", "research scientist"], "15-2041.00": ["biostatistician", "statistical analyst"], "15-2051.01": ["business intelligence analyst"],
 "15-2031.00": ["operations research analyst"], "15-1243.00": ["database architect"], "15-1242.00": ["etl developer", "database administrator"],
 "15-1241.00": ["cloud engineer", "solutions architect"], "15-1299.08": ["site reliability engineer", "cloud engineer"], "15-1244.00": ["system admin", "network administrator"],
 "11-3021.00": ["it infrastructure manager", "it head"], "15-1254.00": ["frontend developer", "react developer"], "15-1253.00": ["software tester", "test engineer"],
 "15-1251.00": ["programmer analyst"], "15-1252.00": ["software engineer"], "15-1211.00": ["system analyst", "business systems analyst"],
 "13-1111.00": ["business consultant"], "13-1082.00": ["program manager", "scrum master"], "13-1161.01": ["seo", "digital marketing executive"]}
# Countries and cities offered in the location dropdowns: [value Adzuna searches, label shown]. js/app.js keeps an identical copy (checked by test_app.py).
LOCATIONS = {"in": {"name": "India", "cities": [["Chennai", "Chennai"], ["Bangalore", "Bengaluru"], ["Hyderabad", "Hyderabad"], ["Pune", "Pune"], ["Mumbai", "Mumbai"], ["Delhi", "Delhi"], ["Noida", "Noida"], ["Gurgaon", "Gurugram"], ["Kolkata", "Kolkata"], ["Coimbatore", "Coimbatore"], ["Kochi", "Kochi"], ["Ahmedabad", "Ahmedabad"], ["Jaipur", "Jaipur"], ["Chandigarh", "Chandigarh"], ["Trivandrum", "Thiruvananthapuram"], ["Madurai", "Madurai"], ["Visakhapatnam", "Visakhapatnam"], ["Bhubaneswar", "Bhubaneswar"], ["Indore", "Indore"]]}, "us": {"name": "United States", "cities": [["New York", "New York"], ["San Francisco", "San Francisco"], ["Seattle", "Seattle"], ["Austin", "Austin"], ["Boston", "Boston"], ["Chicago", "Chicago"], ["Los Angeles", "Los Angeles"], ["Washington", "Washington, DC"], ["Atlanta", "Atlanta"], ["Dallas", "Dallas"]]}, "gb": {"name": "United Kingdom", "cities": [["London", "London"], ["Manchester", "Manchester"], ["Birmingham", "Birmingham"], ["Edinburgh", "Edinburgh"], ["Glasgow", "Glasgow"], ["Bristol", "Bristol"], ["Leeds", "Leeds"], ["Cambridge", "Cambridge"]]}, "sg": {"name": "Singapore", "cities": []}, "au": {"name": "Australia", "cities": [["Sydney", "Sydney"], ["Melbourne", "Melbourne"], ["Brisbane", "Brisbane"], ["Perth", "Perth"], ["Adelaide", "Adelaide"], ["Canberra", "Canberra"]]}, "ca": {"name": "Canada", "cities": [["Toronto", "Toronto"], ["Vancouver", "Vancouver"], ["Montreal", "Montreal"], ["Ottawa", "Ottawa"], ["Calgary", "Calgary"], ["Waterloo", "Waterloo"]]}}
COUNTRY = {"in": "India", "us": "the United States", "gb": "the United Kingdom", "sg": "Singapore", "au": "Australia", "ca": "Canada"}
CURRENCY = {"in": "₹", "us": "$", "gb": "£", "au": "A$", "ca": "C$", "sg": "S$", "de": "€", "fr": "€", "nl": "€"}

class LiveJobsError(Exception):
    pass

def _cached(key, fn):
    hit = CACHE.get(key)
    if hit and time.time() - hit[0] < TTL: return hit[1]
    val = fn(); CACHE[key] = (time.time(), val); return val

def _clean(text):
    return re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", " ", text or ""))).strip()

def adzuna(occ, country="in", where="", n=20):
    app_id, app_key = os.environ.get("ADZUNA_APP_ID"), os.environ.get("ADZUNA_APP_KEY")
    if not app_id or not app_key:
        raise LiveJobsError("Live search needs a free Adzuna key. Add ADZUNA_APP_ID and ADZUNA_APP_KEY to the .env file (see README).")
    what = QUERY.get(occ)
    if not what: raise LiveJobsError("Unknown occupation.")
    def call(what=what, where=where):
        r = requests.get(f"https://api.adzuna.com/v1/api/jobs/{country}/search/1", timeout=15,
                         params={"app_id": app_id, "app_key": app_key, "what": what, "where": where, "results_per_page": n,
                                 "max_days_old": 60, "sort_by": "date", "content-type": "application/json"})
        if r.status_code in (401, 403): raise LiveJobsError("Adzuna rejected the key. Check ADZUNA_APP_ID and ADZUNA_APP_KEY.")
        r.raise_for_status()
        cur = CURRENCY.get(country, "")
        out = []
        for j in r.json().get("results", []):
            lo, hi = j.get("salary_min"), j.get("salary_max")
            if lo is not None and hi is not None:
                sal = f"{cur}{int(lo):,}" + (f"–{cur}{int(hi):,}" if hi != lo else "")
            elif lo is not None:
                sal = f"From {cur}{int(lo):,}"
            elif hi is not None:
                sal = f"Up to {cur}{int(hi):,}"
            else:
                sal = ""
            out.append(dict(title=_clean(j.get("title")), company=(j.get("company") or {}).get("display_name", "Company not listed"),
                            location=(j.get("location") or {}).get("display_name", ""), salary=sal,
                            date=(j.get("created") or "")[:10], url=j.get("redirect_url", ""), remote="remote" in (j.get("title") or "").lower(),
                            text=_clean(j.get("title", "")) + " " + _clean(j.get("description", ""))))
        return out
    # 1) title in the city  2) related titles in the city  3) title country-wide  4) related titles country-wide (max 4 API calls)
    attempts = [(what, where)] + [(a, where) for a in ALT.get(occ, [])[:1]]
    if where: attempts += [(what, "")] + [(a, "") for a in ALT.get(occ, [])[:1]]
    for k, (w, loc) in enumerate(attempts):
        jobs = _cached(("adzuna", w, country, loc.lower(), n), lambda w=w, loc=loc: call(w, loc))
        if jobs:
            note = ""
            if k:
                place = f"in {loc}" if loc else f"across {COUNTRY.get(country, country.upper())}"
                why = f"No “{what}” openings near {where}; " if where and loc != where else (f"No “{what}” openings; " if w != what else "")
                note = why + (f"showing “{w}” openings {place}." if w != what else f"showing openings {place}.")
            return dict(source="Adzuna", attribution="Jobs by Adzuna", jobs=jobs, note=note)
    return dict(source="Adzuna", attribution="Jobs by Adzuna", jobs=[],
                note=f"No current openings for “{what}” or related titles in {COUNTRY.get(country, country.upper())}. Try another country or the 2024 dataset tab.")

def remotive(occ, n=20):
    what = QUERY.get(occ)
    if not what: raise LiveJobsError("Unknown occupation.")
    def feed():
        r = requests.get("https://remotive.com/api/remote-jobs", params={"search": what}, timeout=20); r.raise_for_status()
        return r.json().get("jobs", [])
    jobs = _cached(("remotive-feed", what), feed)
    words = what.split()
    keep = [j for j in jobs if all(w in (j.get("title") or "").lower() for w in words)]   # filter ourselves: server-side search is unreliable
    out = [dict(title=_clean(j.get("title")), company=j.get("company_name") or j.get("company") or "Company not listed",
                location=j.get("candidate_required_location") or "Remote", salary=j.get("salary") or "",
                date=(j.get("publication_date") or "")[:10], url=j.get("url", ""), remote=True,
                text=_clean(j.get("title", "")) + " " + _clean(j.get("description", ""))) for j in keep[:n]]
    return dict(source="Remotive", attribution="Remote jobs from Remotive.com (listings are delayed by 24 hours)", jobs=out)
