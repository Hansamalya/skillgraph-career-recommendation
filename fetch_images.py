"""
fetch_images.py — download the occupation, hero and login photos once, so the apps show real photos offline.

    python fetch_images.py            (skips photos already downloaded; add --force to re-download)

Photos come from Unsplash (free to use under the Unsplash licence; credit "Photos: Unsplash", as the site footer does).
Saved to assets/img/<O*NET code>.jpg, assets/img/hero.jpg and assets/img/login.jpg.
If a photo cannot be downloaded, the apps fall back to the built-in domain banners in assets/banners/.
"""
import json, sys
from pathlib import Path
import requests

ROOT = Path(__file__).resolve().parent; OUT = ROOT / "assets" / "img"; OUT.mkdir(parents=True, exist_ok=True)
U = "https://images.unsplash.com/photo-{}?auto=format&fit=crop&w=1200&q=75"
EXTRA = {"hero": U.format("1522071820081-009f0129c71c"), "login": U.format("1523240795612-9a054b0db644")}

def main():
    force = "--force" in sys.argv
    occ = json.loads((ROOT / "data" / "kg_web.json").read_text(encoding="utf-8"))["occupations"]
    jobs = {o["id"]: o["image"] for o in occ if o.get("image")} | EXTRA
    ok, failed = 0, []
    for key, url in jobs.items():
        dest = OUT / f"{key}.jpg"
        if dest.exists() and not force: ok += 1; continue
        try:
            r = requests.get(url, timeout=25, headers={"User-Agent": "SkillGraph student project"})
            r.raise_for_status()
            if not r.headers.get("content-type", "").startswith("image"): raise ValueError("not an image")
            dest.write_bytes(r.content); ok += 1; print(f"  ✓ {key}")
        except Exception as e:
            failed.append(key); print(f"  ✗ {key}: {type(e).__name__} (the app will use the domain banner)")
    print(f"\n{ok} of {len(jobs)} photos available in {OUT}" + (f"; {len(failed)} will use banners" if failed else ""))

if __name__ == "__main__":
    main()
