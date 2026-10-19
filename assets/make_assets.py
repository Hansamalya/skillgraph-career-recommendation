"""make_assets.py — draws the SkillGraph logo and the built-in banners (no internet needed)."""
import math, random
from pathlib import Path
from PIL import Image, ImageDraw, ImageFilter, ImageFont
A = Path(__file__).resolve().parent
INK, ORANGE, TEAL, ROSE = (27, 42, 65), (242, 165, 65), (127, 209, 199), (239, 138, 151)
DOMAINS = {"ai-machine-learning": (46, 158, 79), "data-analytics-bi": (31, 119, 180), "cloud-devops": (140, 86, 75),
           "software-web": (148, 103, 189), "cybersecurity": (214, 39, 40), "ui-ux-design": (194, 84, 157)}

def font(size, bold=True):
    for f in (["DejaVuSans-Bold.ttf", "arialbd.ttf"] if bold else ["DejaVuSans.ttf", "arial.ttf"]):
        try: return ImageFont.truetype(f, size)
        except OSError: pass
    return ImageFont.load_default()

def logo(size=512):
    S = size * 4; im = Image.new("RGBA", (S, S), (0, 0, 0, 0)); d = ImageDraw.Draw(im)
    d.ellipse([0, 0, S - 1, S - 1], fill=INK)
    top, left, right = (S * .5, S * .29), (S * .29, S * .68), (S * .71, S * .68)
    for a, b in [(top, left), (top, right), (left, right)]: d.line([a, b], fill=(255, 255, 255, 150), width=int(S * .028))
    for (x, y), c, r in [(top, ORANGE, .105), (left, TEAL, .09), (right, ROSE, .09)]:
        rr = S * r; d.ellipse([x - rr, y - rr, x + rr, y + rr], fill=c, outline=INK, width=int(S * .02))
    return im.resize((size, size), Image.LANCZOS)

def wordmark():
    lg = logo(160); W, H = 760, 200
    im = Image.new("RGBA", (W, H), (0, 0, 0, 0)); im.paste(lg, (20, 20), lg)
    ImageDraw.Draw(im).text((200, 52), "SkillGraph", font=font(84), fill=INK)
    return im

def network(W, H, base, seed, accents, n=46, text=None):
    rnd = random.Random(seed); im = Image.new("RGB", (W, H)); px = im.load()
    for x in range(W):                                   # diagonal gradient: domain colour → ink
        t = x / W
        col = tuple(int(base[i] * (1 - t) * .9 + INK[i] * (t * .9 + .1)) for i in range(3))
        for y in range(H): px[x, y] = col
    ov = Image.new("RGBA", (W, H), (0, 0, 0, 0)); d = ImageDraw.Draw(ov)
    pts = [(rnd.uniform(.35, 1.02) * W, rnd.uniform(-.05, 1.05) * H) for _ in range(n)]
    for i, p in enumerate(pts):                          # link each node to its 3 nearest neighbours
        for q in sorted(pts, key=lambda q: math.dist(p, q))[1:4]:
            d.line([p, q], fill=(255, 255, 255, 55), width=2)
    for i, (x, y) in enumerate(pts):
        r = rnd.choice([5, 7, 9, 12]); c = rnd.choice(accents) if i % 3 == 0 else (255, 255, 255)
        d.ellipse([x - r, y - r, x + r, y + r], fill=c + (230 if i % 3 == 0 else 140,))
    glow = ov.filter(ImageFilter.GaussianBlur(3)); im = Image.alpha_composite(im.convert("RGBA"), glow); im = Image.alpha_composite(im, ov)
    if text:
        ImageDraw.Draw(im).text((48, H - 92), text, font=font(46), fill=(255, 255, 255, 235))
    return im.convert("RGB")

if __name__ == "__main__":
    logo(512).save(A / "logo.png"); logo(64).save(A / "favicon.png"); wordmark().save(A / "logo_wordmark.png")
    (A / "logo.svg").write_text('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 32 32"><circle cx="16" cy="16" r="15.5" fill="#1b2a41"/>'
        '<path d="M16 9.3 L9.3 21.8 L22.7 21.8 Z" fill="none" stroke="#fff" stroke-opacity=".6" stroke-width=".9"/>'
        '<circle cx="16" cy="9.3" r="3.4" fill="#f2a541"/><circle cx="9.3" cy="21.8" r="2.9" fill="#7fd1c7"/><circle cx="22.7" cy="21.8" r="2.9" fill="#ef8a97"/></svg>')
    network(1600, 640, (31, 56, 100), 7, [ORANGE, TEAL, ROSE] + list(DOMAINS.values()), n=70).save(A / "banners" / "hero.jpg", quality=88)
    names = {"ai-machine-learning": "AI & Machine Learning", "data-analytics-bi": "Data Analytics & BI", "cloud-devops": "Cloud & DevOps",
             "software-web": "Software & Web", "cybersecurity": "Cybersecurity", "ui-ux-design": "UI/UX & Design"}
    for k, (slug, col) in enumerate(DOMAINS.items()):
        network(1200, 520, col, 11 + k, [ORANGE, TEAL, ROSE]).save(A / "banners" / f"{slug}.jpg", quality=88)
        network(1200, 520, col, 11 + k, [ORANGE, TEAL, ROSE], text=names[slug]).save(A / "banners" / f"{slug}_titled.jpg", quality=88)
    print("assets written to", A)
