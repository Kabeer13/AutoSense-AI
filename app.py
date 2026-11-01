"""
AUTOSENSE AI — Intelligence behind every engine.

A cinematic, luxury-automotive front end for the AutoSense AI engine-fault
diagnosis research project.

Run with:   streamlit run app.py

Project layout expected next to this file
    data/EngineFaultDB_Final.csv
    models/pca_lda.joblib | lda_only.joblib | random_forest.joblib
    results/model_metrics.json
    results/sensor_reliability_results.csv
    assets/                       (optional) hero.jpg, intelligence.jpg,
                                  engine.jpg, reliability.jpg, philosophy.jpg

Imagery
    No image URLs are invented. Every cinematic section ships with an original,
    code-drawn fallback artwork. To use photography, either drop licensed files
    into ./assets (names above; .jpg/.jpeg/.png/.webp, keep each under ~600 KB)
    or paste direct image URLs you have the rights to into IMAGES below.
"""

from __future__ import annotations

import base64
import html as _html
import inspect
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import quote

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# =========================================================
# CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="AutoSense AI | Intelligence Behind Every Engine",
    page_icon="◆",
    layout="wide",
    initial_sidebar_state="collapsed",
)

BASE_DIR = Path(__file__).resolve().parent

DATA_PATH = BASE_DIR / "data" / "EngineFaultDB_Final.csv"
METRICS_PATH = BASE_DIR / "results" / "model_metrics.json"
RELIABILITY_PATH = BASE_DIR / "results" / "sensor_reliability_results.csv"
ASSET_DIR = BASE_DIR / "assets"

MODEL_PATHS = {
    "PCA + LDA": BASE_DIR / "models" / "pca_lda.joblib",
    "LDA Only": BASE_DIR / "models" / "lda_only.joblib",
    "Random Forest": BASE_DIR / "models" / "random_forest.joblib",
}

FAULT_LABELS = {
    0: "Normal",
    1: "Rich Mixture",
    2: "Lean Mixture",
    3: "Low Ignition Voltage",
}

FAULT_NOTES = {
    0: "Sensor readings resemble the normal operating condition in the training data.",
    1: "Sensor readings resemble the rich-mixture condition: more fuel relative to air than ideal.",
    2: "Sensor readings resemble the lean-mixture condition: less fuel relative to air than ideal.",
    3: "Sensor readings resemble the low-ignition-voltage condition: a weak supply to the ignition system.",
}

PAGES = {
    "home": "Home",
    "diagnostics": "Engine Diagnostics",
    "models": "Model Intelligence",
    "reliability": "Sensor Reliability",
    "research": "Research",
}

# Optional direct image URLs (leave None to use ./assets or the built-in artwork).
IMAGES: dict[str, str | None] = {
    "hero": None,
    "intelligence": None,
    "engine": None,
    "reliability": None,
    "philosophy": None,
}

GOLD = "#D9BC7E"
GOLD_DEEP = "#B79A5E"
BRONZE = "#8E6B3E"
IVORY = "#F3EDE0"
MUTED = "#9C927F"
SLATE = "#6F8A9B"

MODEL_COLORS = {
    "PCA + LDA": "#8E6B3E",
    "LDA Only": "#B9B1A0",
    "Random Forest": "#E3C987",
}

MODEL_BLURBS = {
    "PCA + LDA": (
        "Reduce, then discriminate.",
        "Principal Component Analysis compresses the sensor channels before "
        "Linear Discriminant Analysis separates the engine conditions.",
    ),
    "LDA Only": (
        "Discriminate directly.",
        "Linear Discriminant Analysis classifies engine conditions straight "
        "from the standardized sensor readings.",
    ),
    "Random Forest": (
        "An ensemble of decision trees.",
        "Many decision trees vote on the engine condition from the sensor "
        "measurements, capturing non-linear relationships.",
    ),
}

SENSOR_GROUPS = [
    ("Engine and load", ["MAP", "TPS", "Force", "Power", "RPM"]),
    ("Fuel and motion", ["Consumption L/H", "Consumption L/100KM", "Speed"]),
    ("Exhaust and mixture", ["CO", "HC", "CO2", "O2", "Lambda", "AFR"]),
]

EXPERIMENT_NOISE = "Gaussian sensor noise"
EXPERIMENT_MISSING = "Missing sensor"

# =========================================================
# STREAMLIT VERSION COMPATIBILITY
# =========================================================


def _st_version() -> tuple[int, int]:
    try:
        major, minor = st.__version__.split(".")[:2]
        return int(major), int(minor)
    except Exception:
        return (1, 0)


def _plotly_supports_width() -> bool:
    try:
        return "width" in inspect.signature(st.plotly_chart).parameters
    except Exception:
        return False


_WIDTH_API = _st_version() >= (1, 50) and _plotly_supports_width()
_PLOTLY_CONFIG = {
    "displaylogo": False,
    "displayModeBar": "hover",
    "modeBarButtonsToRemove": ["lasso2d", "select2d", "autoScale2d"],
}


def show_chart(fig: go.Figure, key: str) -> None:
    if _WIDTH_API:
        st.plotly_chart(fig, width="stretch", key=key, config=_PLOTLY_CONFIG)
    else:
        st.plotly_chart(fig, use_container_width=True, key=key, config=_PLOTLY_CONFIG)


# =========================================================
# SMALL HELPERS
# =========================================================


def esc(value) -> str:
    return _html.escape(str(value), quote=True)


def H(markup: str) -> str:
    """Strip indentation and blank lines so Markdown never treats HTML as code."""
    return "\n".join(line.strip() for line in markup.strip().splitlines() if line.strip())


def md(markup: str) -> None:
    st.markdown(H(markup), unsafe_allow_html=True)


def pct(x: float, digits: int = 2) -> str:
    return f"{x * 100:.{digits}f}"


def signed(points: float, digits: int = 2) -> str:
    text = f"{points:+.{digits}f}"
    return text.replace("-", "−")


def pts(points: float, digits: int = 2) -> str:
    """Points LOST: positive = performance lost, negative = slightly better."""
    return f"{points:.{digits}f}".replace("-", "\u2212")


def lose(points: float, digits: int = 2) -> str:
    """Format an accuracy LOSS (positive = worse) with a typographic minus."""
    return signed(-points, digits)


# =========================================================
# ORIGINAL FALLBACK ARTWORK (drawn in code, no external assets)
# =========================================================


def _svg_uri(svg: str) -> str:
    return "data:image/svg+xml;utf8," + quote(svg)


def art_tach(cx: float, cy: float) -> str:
    parts = [
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1600 900" '
        'preserveAspectRatio="xMidYMid slice">',
        '<defs><radialGradient id="g" cx="50%" cy="50%" r="50%">'
        '<stop offset="0" stop-color="#2a2114" stop-opacity=".95"/>'
        '<stop offset="1" stop-color="#0a0a09" stop-opacity="0"/></radialGradient></defs>',
        '<rect width="1600" height="900" fill="#0a0a09"/>',
        f'<circle cx="{cx}" cy="{cy}" r="560" fill="url(#g)"/>',
    ]
    for r, op in [(110, .42), (200, .30), (290, .22), (380, .16), (470, .11)]:
        parts.append(
            f'<circle cx="{cx}" cy="{cy}" r="{r}" fill="none" stroke="{GOLD}" '
            f'stroke-opacity="{op}" stroke-width="1"/>'
        )
    for i in range(0, 91):
        ang = math.radians(135 + i * 3)
        major = i % 5 == 0
        r1, r2 = 520, (566 if major else 546)
        x1, y1 = cx + r1 * math.cos(ang), cy + r1 * math.sin(ang)
        x2, y2 = cx + r2 * math.cos(ang), cy + r2 * math.sin(ang)
        parts.append(
            f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
            f'stroke="{GOLD}" stroke-opacity="{.5 if major else .22}" stroke-width="{1.6 if major else 1}"/>'
        )
    ang = math.radians(300)
    parts.append(
        f'<line x1="{cx}" y1="{cy}" x2="{cx + 500 * math.cos(ang):.1f}" '
        f'y2="{cy + 500 * math.sin(ang):.1f}" stroke="{GOLD}" stroke-opacity=".55" stroke-width="1.4"/>'
    )
    parts.append(f'<circle cx="{cx}" cy="{cy}" r="7" fill="{GOLD}" fill-opacity=".8"/>')
    parts.append("</svg>")
    return _svg_uri("".join(parts))


def art_terrain(phase: float) -> str:
    parts = [
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1600 900" '
        'preserveAspectRatio="xMidYMid slice">',
        '<rect width="1600" height="900" fill="#0a0a09"/>',
    ]
    for k in range(0, 30):
        y0 = 90 + k * 28
        pts = []
        for x in range(0, 1681, 40):
            y = (
                y0
                + 42 * math.sin(x / 190 + k * 0.33 + phase)
                + 24 * math.sin(x / 71 + k * 0.81 + phase * 1.7)
            )
            pts.append(f"{x},{y:.1f}")
        op = 0.07 + 0.2 * (1 - abs(k - 14) / 15)
        parts.append(
            f'<polyline points="{" ".join(pts)}" fill="none" stroke="{GOLD}" '
            f'stroke-opacity="{op:.2f}" stroke-width="1"/>'
        )
    parts.append("</svg>")
    return _svg_uri("".join(parts))


def art_engine() -> str:
    """Schematic four-cylinder engine block, drawn on a square canvas so all cylinders survive cropping."""
    parts = [
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1000 1000" '
        'preserveAspectRatio="xMidYMid slice">',
        '<rect width="1000" height="1000" fill="#0c0b09"/>',
        '<defs><radialGradient id="e" cx="50%" cy="45%" r="60%"><stop offset="0" stop-color="#241d11" stop-opacity=".9"/>'
        '<stop offset="1" stop-color="#0c0b09" stop-opacity="0"/></radialGradient></defs>',
        '<rect width="1000" height="1000" fill="url(#e)"/>',
        f'<rect x="60" y="170" width="880" height="620" rx="8" fill="none" stroke="{GOLD}" stroke-opacity=".3"/>',
        f'<rect x="60" y="130" width="880" height="40" fill="none" stroke="{GOLD}" stroke-opacity=".22"/>',
    ]
    for i in range(4):
        x = 100 + i * 205
        lift = 70 if i in (1, 2) else 0
        parts.append(
            f'<rect x="{x}" y="210" width="165" height="470" rx="12" fill="none" stroke="{GOLD}" stroke-opacity=".42"/>'
        )
        parts.append(
            f'<circle cx="{x + 82}" cy="250" r="14" fill="none" stroke="{GOLD}" stroke-opacity=".7"/>'
        )
        parts.append(
            f'<rect x="{x + 18}" y="{300 + lift}" width="129" height="92" rx="5" fill="none" stroke="{GOLD}" stroke-opacity=".6"/>'
        )
        parts.append(
            f'<line x1="{x + 82}" y1="{392 + lift}" x2="{x + 82}" y2="800" stroke="{GOLD}" stroke-opacity=".4"/>'
        )
        parts.append(
            f'<circle cx="{x + 82}" cy="800" r="38" fill="none" stroke="{GOLD}" stroke-opacity=".55"/>'
        )
        parts.append(
            f'<circle cx="{x + 82}" cy="800" r="9" fill="{GOLD}" fill-opacity=".5"/>'
        )
    parts.append(
        f'<line x1="60" y1="800" x2="940" y2="800" stroke="{GOLD}" stroke-opacity=".3" stroke-dasharray="4 9"/>'
    )
    parts.append("</svg>")
    return _svg_uri("".join(parts))


ART_FOR = {
    "hero": lambda: art_tach(1180, 470),
    "intelligence": lambda: art_terrain(0.0),
    "engine": art_engine,
    "reliability": lambda: art_terrain(2.3),
    "philosophy": lambda: art_tach(420, 450),
}

OVERLAYS = {
    "hero": (
        "linear-gradient(180deg, rgba(5,5,5,.60) 0%, rgba(5,5,5,.10) 34%, rgba(5,5,5,.92) 100%), "
        "linear-gradient(90deg, rgba(5,5,5,.82) 0%, rgba(5,5,5,.10) 66%)"
    ),
    "intelligence": (
        "linear-gradient(90deg, rgba(5,5,5,.88) 0%, rgba(5,5,5,.45) 55%, rgba(5,5,5,.75) 100%)"
    ),
    "engine": "linear-gradient(180deg, rgba(5,5,5,.15), rgba(5,5,5,.55))",
    "reliability": (
        "linear-gradient(180deg, rgba(10,10,9,.55) 0%, rgba(10,10,9,.35) 45%, #0a0a09 100%)"
    ),
    "philosophy": (
        "linear-gradient(180deg, rgba(10,10,9,.75) 0%, rgba(10,10,9,.4) 50%, rgba(10,10,9,.88) 100%)"
    ),
}


@st.cache_data(show_spinner=False)
def _local_asset_uri(path_str: str, mtime: float) -> str | None:
    path = Path(path_str)
    try:
        if path.stat().st_size > 6 * 1024 * 1024:
            return None
        mime = {
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".png": "image/png",
            ".webp": "image/webp",
        }[path.suffix.lower()]
        return f"data:{mime};base64," + base64.b64encode(path.read_bytes()).decode()
    except Exception:
        return None


def image_source(key: str) -> str | None:
    url = IMAGES.get(key)
    if url:
        return url
    for ext in (".jpg", ".jpeg", ".png", ".webp"):
        candidate = ASSET_DIR / f"{key}{ext}"
        if candidate.exists():
            uri = _local_asset_uri(str(candidate), candidate.stat().st_mtime)
            if uri:
                return uri
    return None


@st.cache_data(show_spinner=False)
def _art_uri(key: str) -> str:
    return ART_FOR[key]()


def image_css() -> str:
    rules = []
    for key, overlay in OVERLAYS.items():
        src = image_source(key)
        layer = src if src else _art_uri(key)
        if src:
            # Photography needs a little more weight under the type.
            overlay = overlay.replace(".15", ".28") if key == "engine" else overlay
        rules.append(
            f'.img-{key}{{background-image:{overlay},url("{layer}");'
            "background-color:#0a0a09;background-size:cover;"
            "background-position:center;background-repeat:no-repeat;}"
        )
    return "\n".join(rules)


# =========================================================
# GLOBAL THEME
# =========================================================

BASE_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Jost:wght@300;400;500;600&family=Inter:wght@300;400;500;600&family=Cormorant+Garamond:ital,wght@0,400;0,500;1,400;1,500&display=swap');

:root{
  --gold:#D9BC7E; --gold-deep:#B79A5E; --bronze:#8E6B3E;
  --ivory:#F3EDE0; --text:#CFC6B2; --muted:#9C927F;
  --bg:#0a0a09; --graphite:#12110f; --panel:#161512; --line:rgba(217,188,126,.2);
  --gut:clamp(20px,6vw,110px);
  color-scheme: dark;
}
html,body{scroll-behavior:smooth;background:#0a0a09;}
[data-testid="stMain"],section.main{scroll-behavior:smooth;overflow-x:hidden;}
.stApp,[data-testid="stAppViewContainer"]{background:#0a0a09 !important;color:var(--text);font-family:'Inter',sans-serif;}

header[data-testid="stHeader"],[data-testid="stToolbar"],[data-testid="stDecoration"],
[data-testid="stSidebar"],[data-testid="collapsedControl"],[data-testid="stSidebarCollapsedControl"],
#MainMenu,footer{display:none !important;}

.block-container,[data-testid="stMainBlockContainer"]{
  padding:0 var(--gut) 0 var(--gut) !important;max-width:100% !important;}
.block-container > [data-testid="stVerticalBlock"],
[data-testid="stMainBlockContainer"] > [data-testid="stVerticalBlock"]{gap:0 !important;}
[data-testid="stElementContainer"]:has(iframe[height="0"]),[data-testid="stElementContainer"]:has(iframe[height="1"]),
[data-testid="stElementContainer"]:has([data-testid="stIFrame"]){position:absolute;width:1px;height:1px;overflow:hidden;margin:0;opacity:0;pointer-events:none;}

.stApp [data-testid="stMarkdown"],.stApp .stMarkdown,.stApp [data-testid="stMarkdownContainer"]{margin-bottom:0 !important;}
.stApp p,.stApp li,.stApp label,.stApp span{font-family:'Inter',sans-serif;}
.stApp ::selection{background:rgba(217,188,126,.35);color:#fff;}

/* ---------- layout primitives ---------- */
.bleed{width:100vw;position:relative;left:50%;margin-left:-50vw;}
.stApp .eyebrow{font-family:'Jost',sans-serif;font-weight:500;font-size:.72rem;letter-spacing:.42em;
  text-transform:uppercase;color:var(--gold);margin:0 0 22px 0;}
.stApp .display{font-family:'Jost',sans-serif;font-weight:300;text-transform:uppercase;letter-spacing:.015em;
  line-height:1.02;color:var(--ivory);font-size:clamp(2.5rem,7vw,7rem);margin:0;}
.stApp .display.md{font-size:clamp(2.1rem,4.6vw,4.5rem);}
.stApp .display.sm{font-size:clamp(1.7rem,3vw,3rem);}
.stApp .display em,.stApp .serif em{font-family:'Cormorant Garamond',serif;font-style:italic;font-weight:500;
  text-transform:none;color:var(--gold);letter-spacing:0;}
.stApp .display .gold{color:var(--gold);}
.stApp .lede{font-size:clamp(1rem,1.2vw,1.18rem);line-height:1.8;color:var(--text);font-weight:300;
  max-width:58ch;margin:0;}
.split-copy .display.md{font-size:clamp(2rem,3.4vw,3.5rem);}
.stApp .small{font-size:.82rem;line-height:1.7;color:var(--muted);font-weight:300;margin:0;}
.stApp .rule{height:1px;width:110px;background:linear-gradient(90deg,var(--gold),transparent);margin:30px 0;}
.stApp .rule.full{width:100%;background:linear-gradient(90deg,var(--line),rgba(217,188,126,.04));margin:0;}
.stApp .serif{font-family:'Cormorant Garamond',serif;font-weight:400;font-style:italic;color:var(--ivory);
  line-height:1.25;margin:0;}

.stApp a.btn,.stApp a.btn:visited{display:inline-flex;align-items:center;gap:16px;padding:19px 36px;
  font-family:'Jost',sans-serif;font-weight:500;font-size:.76rem;letter-spacing:.3em;text-transform:uppercase;
  text-decoration:none !important;border-radius:0;transition:all .5s cubic-bezier(.2,.7,.2,1);cursor:pointer;}
.stApp a.btn.gold{background:linear-gradient(110deg,#B79A5E 0%,#E8D4A2 50%,#B79A5E 100%);background-size:220% 100%;
  background-position:0 0;color:#15110a !important;}
.stApp a.btn.gold:hover{background-position:100% 0;transform:translateY(-3px);box-shadow:0 18px 44px rgba(217,188,126,.28);}
.stApp a.btn.ghost{border:1px solid rgba(217,188,126,.6);color:var(--ivory) !important;background:transparent;}
.stApp a.btn.ghost:hover{background:rgba(217,188,126,.12);border-color:var(--gold);transform:translateY(-3px);}
.stApp a.btn .arr{display:inline-block;width:26px;height:1px;background:currentColor;position:relative;transition:width .4s;}
.stApp a.btn .arr::after{content:"";position:absolute;right:0;top:-3px;width:7px;height:7px;border-top:1px solid currentColor;
  border-right:1px solid currentColor;transform:rotate(45deg);}
.stApp a.btn:hover .arr{width:40px;}
.btn-row{display:flex;flex-wrap:wrap;gap:18px;margin-top:44px;}

/* ---------- navigation ---------- */
.as-nav{position:fixed;top:0;left:0;right:0;z-index:99999;display:flex;align-items:center;justify-content:space-between;
  height:88px;padding:0 var(--gut);border-bottom:1px solid transparent;
  background:linear-gradient(180deg,rgba(6,6,6,.82),rgba(6,6,6,0));
  transition:background .5s,height .4s,border-color .5s,backdrop-filter .5s;}
.as-nav.solid,.as-nav.always{background:rgba(8,8,7,.94);backdrop-filter:blur(16px);-webkit-backdrop-filter:blur(16px);
  height:70px;border-bottom:1px solid rgba(217,188,126,.18);}
.stApp .as-nav a.brand{display:flex;align-items:center;gap:14px;text-decoration:none !important;}
.as-nav .mark{width:18px;height:18px;border:1px solid var(--gold);transform:rotate(45deg);position:relative;flex:none;
  transition:transform .8s cubic-bezier(.2,.7,.2,1);}
.as-nav .mark::after{content:"";position:absolute;inset:4px;background:var(--gold);}
.stApp .as-nav a.brand:hover .mark{transform:rotate(225deg);}
.as-nav .wm{font-family:'Jost',sans-serif;font-weight:500;font-size:1.02rem;letter-spacing:.34em;color:var(--ivory);}
.as-nav .wm i{font-style:normal;color:var(--gold);margin-left:.5em;font-weight:400;}
.as-nav .nav-links{display:flex;gap:clamp(18px,2.8vw,46px);align-items:center;}
.stApp .as-nav a.nav-link{position:relative;font-family:'Jost',sans-serif;font-weight:400;font-size:.7rem;letter-spacing:.28em;
  text-transform:uppercase;color:#E6DCC6 !important;text-decoration:none !important;padding:10px 0;transition:color .3s;}
.as-nav a.nav-link::after{content:"";position:absolute;left:0;right:0;bottom:2px;height:1px;background:var(--gold);
  transform:scaleX(0);transform-origin:left;transition:transform .5s cubic-bezier(.2,.7,.2,1);}
.stApp .as-nav a.nav-link:hover{color:var(--gold) !important;}
.as-nav a.nav-link:hover::after,.as-nav a.nav-link.active::after{transform:scaleX(1);}
.stApp .as-nav a.nav-link.active{color:var(--gold) !important;}
@media (max-width:900px){
  .as-nav,.as-nav.solid,.as-nav.always{flex-direction:column;justify-content:center;height:auto;padding:12px 16px 8px;gap:2px;
    background:rgba(8,8,7,.95);}
  .as-nav .nav-links{gap:16px;overflow-x:auto;max-width:100%;}
  .stApp .as-nav a.nav-link{font-size:.6rem;letter-spacing:.18em;white-space:nowrap;}
}

/* ---------- hero ---------- */
.hero{position:relative;min-height:100vh;display:flex;align-items:flex-end;overflow:hidden;}
.hero-bg{position:absolute;inset:-3%;animation:drift 40s ease-in-out infinite alternate;}
.hero::after{content:"";position:absolute;inset:0;pointer-events:none;
  background:radial-gradient(ellipse at 30% 110%,rgba(10,10,9,.9),transparent 60%);}
.hero-in{position:relative;z-index:2;padding:0 var(--gut) clamp(90px,15vh,170px);max-width:1280px;}
.hero .display{font-size:clamp(2.7rem,8.4vw,8.6rem);line-height:.98;}
.rise{animation:rise 1.3s cubic-bezier(.2,.7,.2,1) both;}
.rise.d1{animation-delay:.18s;}.rise.d2{animation-delay:.36s;}.rise.d3{animation-delay:.54s;}.rise.d4{animation-delay:.72s;}
.scroll-ind{position:absolute;z-index:3;left:50%;bottom:26px;transform:translateX(-50%);display:flex;flex-direction:column;
  align-items:center;gap:12px;font-family:'Jost',sans-serif;font-size:.58rem;letter-spacing:.5em;color:var(--muted);
  text-transform:uppercase;}
.scroll-ind .ln{width:1px;height:58px;background:linear-gradient(180deg,var(--gold),transparent);transform-origin:top;
  animation:drip 2.4s ease-in-out infinite;}
.hero-meta{position:absolute;z-index:3;right:var(--gut);bottom:34px;text-align:right;font-family:'Jost',sans-serif;
  font-size:.62rem;letter-spacing:.36em;text-transform:uppercase;color:var(--muted);line-height:2;}
.hero-meta b{color:var(--gold);font-weight:500;}
@media (max-width:900px){.hero-meta{display:none;}}

.phero{position:relative;min-height:60vh;display:flex;align-items:flex-end;overflow:hidden;}
.phero .hero-bg{animation-duration:55s;}
.phero::after{content:"";position:absolute;inset:0;background:linear-gradient(180deg,rgba(10,10,9,0) 40%,#0a0a09 100%);pointer-events:none;}
.phero-in{position:relative;z-index:2;padding:150px var(--gut) clamp(50px,9vh,100px);max-width:1200px;}
.phero .display{font-size:clamp(2.4rem,6vw,5.8rem);}

/* ---------- sections ---------- */
.sec{position:relative;padding:clamp(80px,14vh,170px) var(--gut);scroll-margin-top:70px;}
.sec.tall{min-height:94vh;display:flex;align-items:center;overflow:hidden;}
.sec .bgfill{position:absolute;inset:0;}
.sec-in{position:relative;z-index:2;width:100%;}
.grid-a{display:grid;grid-template-columns:1.25fr 1fr;gap:clamp(40px,7vw,120px);align-items:center;}
@media (max-width:900px){.grid-a{grid-template-columns:1fr;}}
.cond{list-style:none;margin:0;padding:0;border-top:1px solid var(--line);}
.cond li{display:grid;grid-template-columns:64px 1fr;gap:18px;padding:26px 0;border-bottom:1px solid var(--line);
  transition:padding .5s cubic-bezier(.2,.7,.2,1),background .5s;}
.cond li:hover{padding-left:16px;background:linear-gradient(90deg,rgba(217,188,126,.08),transparent);}
.cond .n{font-family:'Jost',sans-serif;font-weight:300;font-size:1.9rem;color:var(--gold);line-height:1;}
.cond .t{font-family:'Jost',sans-serif;font-weight:400;font-size:1.15rem;letter-spacing:.14em;text-transform:uppercase;color:var(--ivory);}
.cond .d{font-size:.88rem;color:var(--muted);margin-top:6px;line-height:1.6;font-weight:300;}

.split{display:grid;grid-template-columns:1.1fr 1fr;min-height:92vh;background:var(--graphite);border-top:1px solid var(--line);
  border-bottom:1px solid var(--line);}
.split-img{position:relative;min-height:56vh;overflow:hidden;}
.split-img .bgfill{position:absolute;inset:0;transition:transform 9s ease;}
.split:hover .split-img .bgfill{transform:scale(1.05);}
.split-img .frame{position:absolute;inset:28px;border:1px solid rgba(217,188,126,.35);pointer-events:none;}
.split-img .cap{position:absolute;left:44px;bottom:42px;font-family:'Jost',sans-serif;font-size:.62rem;letter-spacing:.4em;
  text-transform:uppercase;color:var(--gold);}
.split-copy{display:flex;flex-direction:column;justify-content:center;padding:clamp(48px,8vw,130px) var(--gut) clamp(48px,8vw,130px) clamp(32px,6vw,110px);}
@media (max-width:900px){.split{grid-template-columns:1fr;}.split-copy{padding:60px var(--gut);}}
.chips{display:flex;flex-wrap:wrap;gap:10px;margin-top:32px;}
.chip{font-family:'Jost',sans-serif;font-size:.66rem;letter-spacing:.22em;text-transform:uppercase;color:var(--text);
  border:1px solid rgba(217,188,126,.28);padding:9px 14px;transition:all .35s;}
.chip:hover{border-color:var(--gold);color:var(--gold);background:rgba(217,188,126,.06);}

.model-row{display:grid;grid-template-columns:70px 1.15fr 1.5fr 1fr;gap:clamp(18px,3vw,56px);align-items:center;
  padding:clamp(30px,5vh,56px) 0;border-top:1px solid var(--line);position:relative;transition:background .5s;}
.model-row:last-child{border-bottom:1px solid var(--line);}
.model-row:hover{background:linear-gradient(90deg,rgba(217,188,126,.06),transparent 70%);}
.model-row .idx{font-family:'Jost',sans-serif;font-weight:300;font-size:1rem;letter-spacing:.2em;color:var(--gold);}
.model-row .mname{font-family:'Jost',sans-serif;font-weight:300;text-transform:uppercase;letter-spacing:.02em;
  font-size:clamp(1.7rem,3.2vw,3.2rem);line-height:1.05;color:var(--ivory);}
.model-row .mtag{font-family:'Cormorant Garamond',serif;font-style:italic;font-size:1.25rem;color:var(--gold);margin-top:10px;}
.model-row .mdesc{font-size:.95rem;line-height:1.8;color:var(--text);font-weight:300;}
.mm{display:flex;gap:36px;}
.mm .m{display:flex;flex-direction:column;}
.mm .big{font-family:'Jost',sans-serif;font-weight:300;font-size:clamp(2rem,3.4vw,3.4rem);line-height:1;color:var(--ivory);}
.mm .big sup{font-size:.4em;color:var(--gold);margin-left:3px;vertical-align:super;}
.mm small{font-family:'Jost',sans-serif;font-size:.6rem;letter-spacing:.3em;text-transform:uppercase;color:var(--muted);margin-top:10px;}
.model-row .bar{grid-column:2 / -1;height:1px;background:rgba(217,188,126,.14);position:relative;margin-top:6px;}
.model-row .bar i{position:absolute;left:0;top:-1px;height:3px;background:linear-gradient(90deg,var(--bronze),var(--gold));}
.model-row.lead .mname{color:var(--gold);}
@media (max-width:1000px){.model-row{grid-template-columns:44px 1fr;}.model-row .mdesc,.model-row .mm{grid-column:2;}
  .model-row .bar{grid-column:1 / -1;}}

.xwrap{overflow-x:auto;margin-top:56px;}
.xgrid{display:grid;grid-template-columns:1.1fr repeat(5,1fr);min-width:860px;border-top:1px solid var(--line);}
.xgrid > div{padding:26px 18px;border-bottom:1px solid var(--line);}
.xgrid .xh{font-family:'Jost',sans-serif;font-size:.62rem;letter-spacing:.3em;text-transform:uppercase;color:var(--gold);}
.xgrid .xm{font-family:'Jost',sans-serif;font-weight:400;font-size:.95rem;letter-spacing:.14em;text-transform:uppercase;color:var(--ivory);}
.xgrid .xv b{display:block;font-family:'Jost',sans-serif;font-weight:300;font-size:clamp(1.4rem,2.2vw,2.1rem);color:var(--ivory);line-height:1.1;}
.xgrid .xv small{display:block;margin-top:8px;font-size:.74rem;color:var(--muted);letter-spacing:.04em;}
.xgrid .xv.bad b{color:#E8D4A2;}
.xgrid .xv.worst b{color:var(--gold);}
.xgrid > div:hover{background:rgba(217,188,126,.04);}

.pillars{display:grid;grid-template-columns:repeat(3,1fr);gap:0;margin-top:80px;border-top:1px solid var(--line);}
.pillars > div{padding:34px 34px 10px 0;}
.pillars .pn{font-family:'Jost',sans-serif;font-size:.64rem;letter-spacing:.4em;color:var(--gold);}
.pillars .pt{font-family:'Jost',sans-serif;font-weight:300;font-size:1.5rem;text-transform:uppercase;letter-spacing:.06em;
  color:var(--ivory);margin:14px 0 12px;}
@media (max-width:900px){.pillars{grid-template-columns:1fr;}}

/* ---------- stat strip / editorial numbers ---------- */
.stats{display:grid;grid-template-columns:repeat(var(--cols,4),1fr);border-top:1px solid var(--line);border-bottom:1px solid var(--line);}
.stats > div{padding:34px 28px 34px 0;}
.stats > div + div{padding-left:28px;border-left:1px solid var(--line);}
.stats .sv{font-family:'Jost',sans-serif;font-weight:300;font-size:clamp(2rem,3.6vw,3.6rem);color:var(--ivory);line-height:1;}
.stats .sv sup{font-size:.4em;color:var(--gold);margin-left:3px;vertical-align:super;}
.stats .sl{font-family:'Jost',sans-serif;font-size:.62rem;letter-spacing:.32em;text-transform:uppercase;color:var(--muted);margin-top:12px;}
@media (max-width:800px){.stats{grid-template-columns:1fr 1fr;}.stats > div:nth-child(3){border-left:none;padding-left:0;}}

/* ---------- research layout ---------- */
.rblock{display:grid;grid-template-columns:minmax(150px,.55fr) 2fr;gap:clamp(24px,5vw,90px);padding:clamp(38px,6vh,64px) 0;border-top:1px solid var(--line);}
.rblock .rn{font-family:'Jost',sans-serif;font-weight:300;font-size:.7rem;letter-spacing:.4em;color:var(--gold);}
.rblock .rt{font-family:'Jost',sans-serif;font-weight:300;font-size:clamp(1.3rem,2vw,1.9rem);text-transform:uppercase;letter-spacing:.05em;
  color:var(--ivory);margin-top:12px;line-height:1.2;}
.rblock .rb p{font-size:1rem;line-height:1.85;color:var(--text);font-weight:300;margin:0 0 14px 0;max-width:74ch;}
.rblock .rb ul{margin:8px 0 0 0;padding:0;list-style:none;}
.rblock .rb li{position:relative;padding:11px 0 11px 28px;border-bottom:1px solid rgba(217,188,126,.1);font-size:.96rem;
  line-height:1.65;color:var(--text);font-weight:300;max-width:74ch;}
.rblock .rb li::before{content:"";position:absolute;left:0;top:21px;width:12px;height:1px;background:var(--gold);}
.stApp .rb a{color:var(--gold) !important;text-decoration:none !important;border-bottom:1px solid rgba(217,188,126,.4);transition:all .3s;}
.stApp .rb a:hover{border-bottom-color:var(--gold);color:var(--ivory) !important;}
.notice{border:1px solid rgba(217,188,126,.35);border-left:3px solid var(--gold);background:rgba(217,188,126,.05);padding:22px 26px;
  font-size:.92rem;line-height:1.75;color:var(--text);font-weight:300;}
.notice b{color:var(--gold);font-weight:500;letter-spacing:.04em;}
@media (max-width:800px){.rblock{grid-template-columns:1fr;gap:12px;}}

/* ---------- diagnostic report ---------- */
.report{border:1px solid rgba(217,188,126,.4);background:linear-gradient(160deg,#1a1710 0%,#0f0e0b 70%);padding:34px 36px;position:relative;
  animation:rise .9s cubic-bezier(.2,.7,.2,1) both;}
.report::before{content:"";position:absolute;left:0;top:0;width:100%;height:2px;background:linear-gradient(90deg,var(--gold),transparent);}
.report .rh{display:flex;justify-content:space-between;gap:12px;flex-wrap:wrap;font-family:'Jost',sans-serif;font-size:.6rem;
  letter-spacing:.34em;text-transform:uppercase;color:var(--muted);}
.report .rh b{color:var(--gold);font-weight:500;}
.report .rl{font-family:'Jost',sans-serif;font-size:.64rem;letter-spacing:.38em;text-transform:uppercase;color:var(--gold);margin-top:34px;}
.report .rr{font-family:'Jost',sans-serif;font-weight:300;text-transform:uppercase;font-size:clamp(2rem,3.6vw,3.5rem);
  line-height:1.05;color:var(--ivory);margin-top:12px;}
.report .rm{display:grid;grid-template-columns:repeat(3,1fr);gap:0;margin-top:30px;border-top:1px solid var(--line);}
.report .rm > div{padding:18px 14px 0 0;}
.report .rm > div + div{padding-left:16px;border-left:1px solid var(--line);}
.report .rm small{display:block;font-family:'Jost',sans-serif;font-size:.56rem;letter-spacing:.3em;text-transform:uppercase;color:var(--muted);}
.report .rm span{display:block;font-family:'Jost',sans-serif;font-weight:300;font-size:1.35rem;color:var(--ivory);margin-top:8px;}
.report .rd{margin:24px 0 0;font-size:.9rem;line-height:1.75;color:var(--text);font-weight:300;}
.idle{border:1px dashed rgba(217,188,126,.3);padding:48px 36px;text-align:left;}
.idle .rl{font-family:'Jost',sans-serif;font-size:.64rem;letter-spacing:.38em;text-transform:uppercase;color:var(--gold);}
.idle .rr{font-family:'Jost',sans-serif;font-weight:300;text-transform:uppercase;font-size:1.8rem;color:var(--ivory);margin-top:14px;line-height:1.2;}
.idle p{margin:16px 0 0;font-size:.92rem;line-height:1.75;color:var(--muted);font-weight:300;}
.fgroup{font-family:'Jost',sans-serif;font-size:.62rem;letter-spacing:.34em;text-transform:uppercase;color:var(--gold);
  padding:4px 0 10px;border-bottom:1px solid var(--line);margin-bottom:6px;}

/* ---------- tables ---------- */
.tbl-wrap{overflow-x:auto;margin:6px 0 4px;}
.stApp table.tbl{width:100%;border-collapse:collapse;font-family:'Inter',sans-serif;background:transparent;}
.stApp table.tbl th{font-family:'Jost',sans-serif;font-weight:500;font-size:.6rem;letter-spacing:.28em;text-transform:uppercase;
  color:var(--gold);text-align:left;padding:14px 16px;border:none;border-bottom:1px solid rgba(217,188,126,.35);background:transparent;white-space:nowrap;}
.stApp table.tbl td{padding:14px 16px;font-size:.88rem;color:var(--text);border:none;border-bottom:1px solid rgba(217,188,126,.1);
  background:transparent;font-weight:300;font-variant-numeric:tabular-nums;}
.stApp table.tbl tr:hover td{background:rgba(217,188,126,.05);color:var(--ivory);}
.stApp table.tbl td.num,.stApp table.tbl th.num{text-align:right;}
.stApp table.tbl td.hl{color:var(--gold);font-weight:500;}
.stApp table.tbl td.gain{color:#8FB0C2;}

/* ---------- footer ---------- */
.foot{background:#070706;border-top:1px solid var(--line);padding:90px var(--gut) 40px;margin-top:clamp(70px,12vh,140px);}
.foot-grid{display:grid;grid-template-columns:1.6fr 1fr 1fr 1.2fr;gap:50px;}
@media (max-width:900px){.foot-grid{grid-template-columns:1fr 1fr;}}
@media (max-width:600px){.foot-grid{grid-template-columns:1fr;}.foot{padding-top:60px;}}
.foot .fb{font-family:'Jost',sans-serif;font-weight:400;font-size:1.5rem;letter-spacing:.34em;color:var(--ivory);}
.foot .fb i{font-style:normal;color:var(--gold);margin-left:.5em;}
.foot .fh{font-family:'Jost',sans-serif;font-size:.62rem;letter-spacing:.36em;text-transform:uppercase;color:var(--gold);margin-bottom:20px;}
.stApp .foot a{display:block;color:var(--text) !important;text-decoration:none !important;font-size:.88rem;font-weight:300;padding:6px 0;transition:all .3s;}
.stApp .foot a:hover{color:var(--gold) !important;padding-left:8px;}
.foot p{font-size:.85rem;line-height:1.8;color:var(--muted);font-weight:300;margin:16px 0 0;max-width:46ch;}
.foot .legal{margin-top:70px;padding-top:26px;border-top:1px solid rgba(217,188,126,.12);display:flex;justify-content:space-between;gap:20px;flex-wrap:wrap;
  font-family:'Jost',sans-serif;font-size:.58rem;letter-spacing:.32em;text-transform:uppercase;color:#6d6555;}

/* ---------- widgets ---------- */
.spacer{height:clamp(30px,6vh,70px);}
.spacer.s{height:22px;}
.stApp [data-testid="stWidgetLabel"] p,.stApp [data-testid="stWidgetLabel"] label,.stApp label p{
  font-family:'Jost',sans-serif !important;font-size:.64rem !important;letter-spacing:.26em !important;text-transform:uppercase !important;
  color:var(--muted) !important;font-weight:500 !important;}
.stApp [data-testid="stNumberInput"] [data-baseweb="input"],.stApp [data-testid="stNumberInput"] [data-baseweb="base-input"]{
  background:#12110e !important;border-radius:0 !important;border-color:#2c271c !important;}
.stApp [data-testid="stNumberInput"] input{color:var(--ivory) !important;font-family:'Inter',sans-serif !important;background:transparent !important;
  -webkit-text-fill-color:var(--ivory) !important;}
.stApp [data-testid="stNumberInput"] [data-baseweb="input"]:focus-within{border-color:var(--gold) !important;box-shadow:0 0 0 1px var(--gold) !important;}
.stApp [data-testid="stNumberInput"] button{background:#1a1812 !important;color:var(--gold) !important;border-color:#2c271c !important;}
.stApp [data-testid="stNumberInput"] button:hover{background:#26221a !important;}
.stApp [data-testid="stSelectbox"] [data-baseweb="select"] > div{background:#12110e !important;border-radius:0 !important;border-color:#2c271c !important;
  color:var(--ivory) !important;}
.stApp [data-testid="stSelectbox"] [data-baseweb="select"] svg{fill:var(--gold) !important;}
.stApp [data-testid="stSelectbox"] [data-baseweb="select"] > div:hover{border-color:var(--gold-deep) !important;}
[data-baseweb="popover"] ul,[data-baseweb="menu"]{background:#14130f !important;border:1px solid rgba(217,188,126,.3);border-radius:0 !important;}
[data-baseweb="popover"] li{color:var(--text) !important;background:transparent !important;}
[data-baseweb="popover"] li:hover,[data-baseweb="popover"] li[aria-selected="true"]{background:rgba(217,188,126,.12) !important;color:var(--gold) !important;}

.stApp div[role="radiogroup"]{gap:10px;flex-wrap:wrap;}
.stApp div[role="radiogroup"] > label{background:transparent;border:1px solid rgba(217,188,126,.28);padding:11px 20px;border-radius:0;
  transition:all .35s;cursor:pointer;margin:0;}
.stApp div[role="radiogroup"] > label > div:first-child{display:none;}
.stApp div[role="radiogroup"] > label:hover{border-color:var(--gold);background:rgba(217,188,126,.06);}
.stApp div[role="radiogroup"] > label p{font-family:'Jost',sans-serif !important;letter-spacing:.2em !important;font-size:.7rem !important;
  text-transform:uppercase;color:var(--text) !important;margin:0 !important;}
.stApp div[role="radiogroup"] > label:has(input:checked){background:linear-gradient(110deg,#B79A5E,#E8D4A2);border-color:transparent;}
.stApp div[role="radiogroup"] > label:has(input:checked) p{color:#15110a !important;font-weight:600 !important;}

.stApp [data-testid="stForm"]{border:1px solid rgba(217,188,126,.22);background:rgba(18,17,14,.7);border-radius:0;padding:30px 30px 26px;}
.stApp [data-testid="stFormSubmitButton"] button,.stApp [data-testid="stFormSubmitButton"] button:focus{
  width:100%;background:linear-gradient(110deg,#B79A5E 0%,#E8D4A2 50%,#B79A5E 100%) !important;background-size:220% 100% !important;
  background-position:0 0;color:#15110a !important;border:none !important;border-radius:0 !important;min-height:58px;
  font-family:'Jost',sans-serif !important;font-weight:600 !important;letter-spacing:.3em !important;text-transform:uppercase;transition:all .5s;}
.stApp [data-testid="stFormSubmitButton"] button p{color:#15110a !important;font-family:'Jost',sans-serif !important;font-weight:600 !important;
  letter-spacing:.3em !important;font-size:.76rem !important;}
.stApp [data-testid="stFormSubmitButton"] button:hover{background-position:100% 0 !important;box-shadow:0 14px 40px rgba(217,188,126,.25);transform:translateY(-2px);}
.stApp [data-testid="stDownloadButton"] button{background:transparent !important;border:1px solid rgba(217,188,126,.6) !important;border-radius:0 !important;
  min-height:52px;padding:0 30px;transition:all .4s;}
.stApp [data-testid="stDownloadButton"] button p{color:var(--ivory) !important;font-family:'Jost',sans-serif !important;letter-spacing:.26em !important;
  font-size:.7rem !important;text-transform:uppercase;}
.stApp [data-testid="stDownloadButton"] button:hover{background:rgba(217,188,126,.12) !important;border-color:var(--gold) !important;}

.stApp [data-baseweb="tab-list"]{gap:30px;border-bottom:1px solid rgba(217,188,126,.18);}
.stApp button[data-baseweb="tab"]{background:transparent !important;padding:14px 0;}
.stApp button[data-baseweb="tab"] p{font-family:'Jost',sans-serif !important;letter-spacing:.26em !important;font-size:.7rem !important;
  text-transform:uppercase;color:var(--muted) !important;}
.stApp button[data-baseweb="tab"][aria-selected="true"] p{color:var(--gold) !important;}
.stApp [data-baseweb="tab-highlight"]{background:var(--gold) !important;height:2px;}
.stApp [data-baseweb="tab-border"]{background:transparent !important;}
.stApp [data-testid="stCaptionContainer"],.stApp [data-testid="stCaptionContainer"] p{color:var(--muted) !important;font-size:.8rem !important;}
.stApp [data-testid="stAlert"]{border-radius:0;background:rgba(217,188,126,.07);border:1px solid rgba(217,188,126,.3);color:var(--text);}
.stApp [data-testid="stPlotlyChart"]{border-top:1px solid rgba(217,188,126,.1);padding-top:6px;}


/* ---------- widgets: current Streamlit (react-aria markup) ---------- */
.stApp [data-testid="stRadioGroup"]{display:flex;flex-wrap:wrap;gap:10px;}
.stApp [data-testid="stRadioGroup"][data-orientation="vertical"]{flex-direction:column;align-items:flex-start;}
.stApp label[data-testid="stRadioOption"]{background:transparent;border:1px solid rgba(217,188,126,.28);padding:11px 20px;border-radius:0;
  transition:all .35s;cursor:pointer;margin:0;}
.stApp label[data-testid="stRadioOption"] > div > div:first-child{display:none;}
.stApp label[data-testid="stRadioOption"]:hover{border-color:var(--gold);background:rgba(217,188,126,.07);}
.stApp label[data-testid="stRadioOption"] p{font-family:'Jost',sans-serif !important;letter-spacing:.2em !important;font-size:.7rem !important;
  text-transform:uppercase;color:var(--text) !important;margin:0 !important;}
.stApp label[data-testid="stRadioOption"][data-selected="true"]{background:linear-gradient(110deg,#B79A5E,#E8D4A2);border-color:transparent;}
.stApp label[data-testid="stRadioOption"][data-selected="true"] p{color:#15110a !important;font-weight:600 !important;}

.stApp [data-testid="stNumberInputContainer"]{background:#12110e !important;border:1px solid #2c271c !important;border-radius:0 !important;
  overflow:hidden;transition:border-color .3s,box-shadow .3s;}
.stApp [data-testid="stNumberInputContainer"]:focus-within{border-color:var(--gold) !important;box-shadow:0 0 0 1px var(--gold);}
.stApp [data-testid="stNumberInputField"]{background:transparent !important;color:var(--ivory) !important;-webkit-text-fill-color:var(--ivory) !important;
  font-family:'Inter',sans-serif !important;}
.stApp [data-testid="stNumberInputStepDown"],.stApp [data-testid="stNumberInputStepUp"]{background:#1a1812 !important;color:var(--gold) !important;
  border:none !important;border-left:1px solid #2c271c !important;border-radius:0 !important;}
.stApp [data-testid="stNumberInputStepDown"]:hover,.stApp [data-testid="stNumberInputStepUp"]:hover{background:#26221a !important;}
.stApp [data-testid="stTooltipIcon"] svg,.stApp [data-testid="stTooltipIcon"] button{color:#6d6555 !important;}

.stApp [data-testid="stSelectbox"] [role="group"],.stApp [data-testid="stSelectbox"] .react-aria-ComboBox > div{
  background:#12110e !important;border:1px solid #2c271c !important;border-radius:0 !important;transition:border-color .3s;}
.stApp [data-testid="stSelectbox"] [role="group"]:hover,.stApp [data-testid="stSelectbox"] [role="group"]:focus-within{border-color:var(--gold) !important;}
.stApp [data-testid="stSelectbox"] input{background:transparent !important;color:var(--ivory) !important;-webkit-text-fill-color:var(--ivory) !important;
  font-family:'Inter',sans-serif !important;}
.stApp [data-testid="stSelectbox"] button{background:transparent !important;color:var(--gold) !important;}
[role="listbox"],.react-aria-Popover [role="listbox"]{background:#14130f !important;border:1px solid rgba(217,188,126,.3) !important;border-radius:0 !important;}
.react-aria-Popover,[data-testid="stSelectboxVirtualDropdown"]{background:#14130f !important;border-radius:0 !important;}
[role="option"]{color:var(--text) !important;background:transparent !important;font-family:'Inter',sans-serif;}
[role="option"][data-focused="true"],[role="option"][data-hovered="true"],[role="option"][aria-selected="true"]{
  background:rgba(217,188,126,.14) !important;color:var(--gold) !important;}

.stApp [data-testid="stTabs"] [role="tablist"]{gap:30px;border-bottom:1px solid rgba(217,188,126,.18);}
.stApp [data-testid="stTabs"] button[role="tab"]{background:transparent !important;padding:14px 0;border-radius:0;}
.stApp [data-testid="stTabs"] button[role="tab"] p{font-family:'Jost',sans-serif !important;letter-spacing:.26em !important;font-size:.7rem !important;
  text-transform:uppercase;color:var(--muted) !important;}
.stApp [data-testid="stTabs"] button[role="tab"][aria-selected="true"] p{color:var(--gold) !important;}
.stApp [data-testid="stTabs"] button[role="tab"][aria-selected="true"]{border-bottom:2px solid var(--gold) !important;color:var(--gold) !important;}
.stApp [data-testid="stTabs"] [data-baseweb="tab-highlight"]{background:var(--gold);}

.stApp [data-testid="stTab"]{background:transparent !important;padding:14px 0 !important;border-radius:0 !important;}
.stApp [data-testid="stTab"] p{font-family:'Jost',sans-serif !important;letter-spacing:.26em !important;font-size:.7rem !important;
  text-transform:uppercase;color:var(--muted) !important;}
.stApp [data-testid="stTab"]:hover p{color:var(--ivory) !important;}
.stApp [data-testid="stTab"][aria-selected="true"] p{color:var(--gold) !important;}
.stApp [data-testid="stTab"] .react-aria-SelectionIndicator{background:var(--gold) !important;height:2px !important;}
.stApp [role="tablist"]{gap:34px !important;border-bottom:1px solid rgba(217,188,126,.18) !important;}

/* ---------- motion ---------- */
@keyframes rise{from{opacity:0;transform:translateY(34px);}to{opacity:1;transform:none;}}
@keyframes drift{from{transform:scale(1.0) translate3d(0,0,0);}to{transform:scale(1.09) translate3d(-1.5%,-1%,0);}}
@keyframes drip{0%{transform:scaleY(0);opacity:0;}30%{opacity:1;}60%{transform:scaleY(1);opacity:1;}100%{transform:scaleY(1);opacity:0;}}
@keyframes fadeUp{from{opacity:0;transform:translateY(22px);}to{opacity:1;transform:none;}}
@supports (animation-timeline: view()){
  .reveal{animation:fadeUp linear both;animation-timeline:view();animation-range:entry 0% entry 60%;}
}
@media (prefers-reduced-motion:reduce){
  .hero-bg,.rise,.reveal,.scroll-ind .ln{animation:none !important;}
  html,body,[data-testid="stMain"]{scroll-behavior:auto;}
}
"""


def inject_css() -> None:
    st.markdown(f"<style>{BASE_CSS}\n{image_css()}</style>", unsafe_allow_html=True)


# Gives the navigation bar its transparent -> solid behaviour on scroll.
NAV_JS = """
<script>
(function () {
  try {
    var P = window.parent, D = P.document;
    function scroller() {
      return D.querySelector('[data-testid="stMain"]') || D.querySelector('section.main') || D.scrollingElement;
    }
    function update() {
      var nav = D.querySelector('.as-nav');
      if (!nav) return;
      var s = scroller();
      var y = Math.max(s ? s.scrollTop : 0, P.scrollY || 0);
      nav.classList.toggle('solid', y > 60);
    }
    P.__asNavUpdate = update;
    if (!P.__asNavBound) {
      D.addEventListener('scroll', function () { P.__asNavUpdate && P.__asNavUpdate(); }, true);
      P.__asNavBound = true;
    }
    update();
  } catch (e) { /* cross-origin embedding: CSS-only navigation still works */ }
})();
</script>
"""

# =========================================================
# DATA LOADING (unchanged project files, read-only)
# =========================================================


@st.cache_data
def load_dataset() -> pd.DataFrame:
    return pd.read_csv(DATA_PATH).drop_duplicates().reset_index(drop=True)


@st.cache_data
def load_metrics() -> dict:
    with open(METRICS_PATH, "r", encoding="utf-8") as handle:
        return json.load(handle)


@st.cache_data
def load_reliability() -> pd.DataFrame:
    return pd.read_csv(RELIABILITY_PATH)


@st.cache_resource
def load_models() -> dict:
    return {name: joblib.load(path) for name, path in MODEL_PATHS.items()}


inject_css()

_required = [DATA_PATH, METRICS_PATH, RELIABILITY_PATH, *MODEL_PATHS.values()]
_missing = [str(p.relative_to(BASE_DIR)) for p in _required if not p.exists()]
if _missing:
    st.error("Required project files are missing: " + ", ".join(_missing))
    st.stop()

df = load_dataset()
metrics = load_metrics()
reliability = load_reliability()
models = load_models()

features = [col for col in df.columns if col != "Fault"]
MODEL_NAMES = list(models.keys())

# ---- derived, data-driven facts (nothing below is typed in by hand) ----


def class_keys(report: dict) -> list[str]:
    return sorted(k for k in report if k.isdigit())


_first_report = next(iter(metrics.values()))["classification_report"]
N_TEST = int(sum(_first_report[k]["support"] for k in class_keys(_first_report)))
N_RECORDS = len(df)
N_TRAIN = N_RECORDS - N_TEST

_clean = reliability[reliability["Condition"] == "Clean"].copy()
_noise = reliability[reliability["Condition"] == "Gaussian Noise"].copy()
_missing_df = reliability[reliability["Condition"].str.startswith("Missing:")].copy()
_missing_df["Sensor"] = _missing_df["Condition"].str.replace("Missing: ", "", regex=False)
_noise["Level"] = (_noise["Severity"] * 100).round().astype(int)
_clean = _clean.assign(Level=0)
NOISE_ALL = pd.concat([_clean, _noise], ignore_index=True).sort_values(["Level", "Model"])
NOISE_LEVELS = sorted(NOISE_ALL["Level"].unique())
SENSORS = list(dict.fromkeys(_missing_df["Sensor"]))


def level_label(level: int) -> str:
    return "Clean" if level == 0 else f"{level}% noise"


# =========================================================
# PLOTLY THEME
# =========================================================


def luxe(fig: go.Figure, height: int = 420, legend: bool = True, margin: dict | None = None) -> go.Figure:
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif", color="#CFC6B2", size=12),
        height=height,
        margin=margin or dict(l=10, r=10, t=36, b=10),
        showlegend=legend,
        legend=dict(orientation="h", y=1.12, x=0, bgcolor="rgba(0,0,0,0)", font=dict(size=12, color="#E6DCC6")),
        hoverlabel=dict(bgcolor="#14130f", bordercolor=GOLD, font=dict(family="Inter", color=IVORY, size=12)),
        colorway=[MODEL_COLORS[m] for m in MODEL_COLORS],
    )
    axis = dict(
        gridcolor="rgba(217,188,126,.09)",
        linecolor="rgba(217,188,126,.28)",
        zeroline=False,
        tickfont=dict(size=11, color="#9C927F"),
        title_font=dict(size=11, color="#9C927F"),
    )
    fig.update_xaxes(**axis)
    fig.update_yaxes(**axis)
    return fig


def html_table(headers: list[str], rows: list[list], num_cols: set[int] | None = None,
               classes: dict | None = None) -> str:
    num_cols = num_cols or set()
    classes = classes or {}
    head = "".join(
        f'<th class="{"num" if i in num_cols else ""}">{esc(h)}</th>' for i, h in enumerate(headers)
    )
    body = []
    for r_i, row in enumerate(rows):
        cells = []
        for c_i, value in enumerate(row):
            cls = ["num"] if c_i in num_cols else []
            extra = classes.get((r_i, c_i))
            if extra:
                cls.append(extra)
            cells.append(f'<td class="{" ".join(cls)}">{esc(value)}</td>')
        body.append("<tr>" + "".join(cells) + "</tr>")
    return (
        '<div class="tbl-wrap"><table class="tbl"><thead><tr>'
        + head + "</tr></thead><tbody>" + "".join(body) + "</tbody></table></div>"
    )


# =========================================================
# SHARED UI BLOCKS
# =========================================================


def mount_script(js: str) -> None:
    """Run a tiny script in the page. Never allowed to break the app."""
    try:
        if hasattr(st, "iframe"):  # current Streamlit
            st.iframe(js, width=1, height=1)
        else:  # older Streamlit
            import streamlit.components.v1 as components

            components.html(js, height=0)
    except Exception:
        pass  # navigation still works with CSS only


def render_nav(active: str) -> None:
    links = "".join(
        f'<a class="nav-link{" active" if key == active else ""}" href="?page={key}" target="_self">{esc(label)}</a>'
        for key, label in PAGES.items()
    )
    always = "" if active == "home" else " always"
    md(
        f"""
        <nav class="as-nav{always}">
        <a class="brand" href="?page=home" target="_self"><span class="mark"></span><span class="wm">AUTOSENSE<i>AI</i></span></a>
        <div class="nav-links">{links}</div>
        </nav>
        """
    )
    mount_script(NAV_JS)


def render_footer() -> None:
    links = "".join(
        f'<a href="?page={key}" target="_self">{esc(label)}</a>' for key, label in PAGES.items()
    )
    md(
        f"""
        <div class="bleed foot">
        <div class="foot-grid">
        <div>
        <div class="fb">AUTOSENSE<i>AI</i></div>
        <p>Engine fault diagnosis, sensor analysis and machine learning, studied together with how dependable a prediction remains when sensors become unreliable.</p>
        </div>
        <div><div class="fh">Explore</div>{links}</div>
        <div><div class="fh">Dataset</div>
        <a href="https://github.com/leoxthomas/EngineFaultDB" target="_blank" rel="noopener">EngineFaultDB on GitHub</a>
        <a href="https://ieeexplore.ieee.org/abstract/document/10311597" target="_blank" rel="noopener">IEEE paper</a>
        </div>
        <div><div class="fh">Notice</div>
        <p style="margin-top:0">An experimental research prototype. Predictions are not a certified mechanical diagnosis and have not been validated beyond the laboratory dataset.</p>
        </div>
        </div>
        <div class="legal"><span>AutoSense AI &nbsp;/&nbsp; Engine Intelligence System</span><span>Research prototype</span></div>
        </div>
        """
    )


def page_hero(image_key: str, eyebrow: str, title_html: str, lede: str) -> None:
    md(
        f"""
        <div class="bleed phero">
        <div class="hero-bg img-{image_key}"></div>
        <div class="phero-in">
        <div class="eyebrow rise">{eyebrow}</div>
        <div class="display rise d1">{title_html}</div>
        <div class="rule rise d2"></div>
        <p class="lede rise d2">{lede}</p>
        </div>
        </div>
        """
    )


def spacer(small: bool = False) -> None:
    md(f'<div class="spacer{" s" if small else ""}"></div>')


def section_heading(eyebrow: str, title_html: str, lede: str = "") -> None:
    lede_html = f'<p class="lede" style="margin-top:22px">{lede}</p>' if lede else ""
    md(
        f"""
        <div class="reveal" style="padding-top:clamp(20px,4vh,50px)">
        <div class="eyebrow">{eyebrow}</div>
        <div class="display sm">{title_html}</div>
        {lede_html}
        </div>
        """
    )


def model_rows_html(show_note: bool = True) -> str:
    best = max(metrics, key=lambda m: metrics[m]["accuracy"])
    rows = []
    for i, name in enumerate(metrics, start=1):
        tag, desc = MODEL_BLURBS.get(name, ("", ""))
        acc, f1 = metrics[name]["accuracy"], metrics[name]["macro_f1"]
        rows.append(
            f"""
            <div class="model-row reveal{" lead" if name == best else ""}">
            <div class="idx">0{i}</div>
            <div><div class="mname">{esc(name)}</div><div class="mtag">{esc(tag)}</div></div>
            <div class="mdesc">{esc(desc)}</div>
            <div class="mm">
            <div class="m"><span class="big">{pct(acc)}<sup>%</sup></span><small>Accuracy</small></div>
            <div class="m"><span class="big">{pct(f1)}<sup>%</sup></span><small>Macro F1</small></div>
            </div>
            <div class="bar"><i style="width:{acc * 100:.2f}%"></i></div>
            </div>
            """
        )
    return H("".join(rows))


# =========================================================
# PAGE: HOME
# =========================================================


def page_home() -> None:
    n_feat = len(features)
    # ---------------- hero ----------------
    md(
        f"""
        <div class="bleed hero">
        <div class="hero-bg img-hero"></div>
        <div class="hero-in">
        <div class="eyebrow rise">AutoSense AI &nbsp;/&nbsp; Engine intelligence</div>
        <div class="display rise d1">Intelligence<br>behind every<br><span class="gold">engine.</span></div>
        <div class="rule rise d2"></div>
        <p class="lede rise d2" style="font-size:clamp(1.05rem,1.5vw,1.4rem);max-width:46ch">Advanced engine fault diagnosis. Intelligent sensor analysis. Precision-driven machine learning.</p>
        <div class="btn-row rise d3">
        <a class="btn gold" href="#engine-intelligence">Explore the system <span class="arr"></span></a>
        <a class="btn ghost" href="?page=diagnostics" target="_self">Run a diagnosis <span class="arr"></span></a>
        </div>
        </div>
        <div class="hero-meta rise d4"><b>{n_feat}</b> sensor channels<br><b>{len(FAULT_LABELS)}</b> engine conditions<br><b>{len(models)}</b> intelligence models</div>
        <div class="scroll-ind"><span>Scroll</span><div class="ln"></div></div>
        </div>
        """
    )

    # ---------------- A · engine intelligence ----------------
    cond_items = "".join(
        f'<li><div class="n">0{code}</div><div><div class="t">{esc(name)}</div>'
        f'<div class="d">{esc(FAULT_NOTES[code])}</div></div></li>'
        for code, name in FAULT_LABELS.items()
    )
    md(
        f"""
        <div class="bleed sec tall" id="engine-intelligence">
        <div class="bgfill img-intelligence"></div>
        <div class="sec-in grid-a">
        <div class="reveal">
        <div class="eyebrow">01 &nbsp;/&nbsp; Engine intelligence</div>
        <div class="display md">Beyond the<br>warning <em>light.</em></div>
        <div class="rule"></div>
        <p class="lede">A warning light tells you that something is wrong. AutoSense AI reads {n_feat} engine sensor channels and uses machine learning to recognise which of four engine conditions the readings most closely resemble.</p>
        </div>
        <ul class="cond reveal">{cond_items}</ul>
        </div>
        </div>
        """
    )

    # ---------------- B · intelligent diagnostics ----------------
    chips = "".join(f'<span class="chip">{esc(f)}</span>' for f in features)
    md(
        f"""
        <div class="bleed split" id="intelligent-diagnostics">
        <div class="split-img"><div class="bgfill img-engine"></div><div class="frame"></div>
        <div class="cap">Engine diagnostics</div></div>
        <div class="split-copy reveal">
        <div class="eyebrow">02 &nbsp;/&nbsp; Intelligent diagnostics</div>
        <div class="display md">One reading.<br>Four possible <em>conditions.</em></div>
        <div class="rule"></div>
        <p class="lede">Choose a model, enter the sensor measurements and receive a predicted engine condition together with the model's output scores for every class, presented as a diagnostic report.</p>
        <div class="chips">{chips}</div>
        <div class="btn-row"><a class="btn gold" href="?page=diagnostics" target="_self">Open diagnostics <span class="arr"></span></a></div>
        </div>
        </div>
        """
    )

    # ---------------- C · three intelligence models ----------------
    md(
        f"""
        <div class="bleed sec" style="padding-bottom:20px">
        <div class="reveal" style="max-width:1100px">
        <div class="eyebrow">03 &nbsp;/&nbsp; Three intelligence models</div>
        <div class="display md">Three ways of <em>reading</em><br>the same engine.</div>
        <div class="rule"></div>
        <p class="lede">Measured on a held-out test set of {N_TEST:,} records. Figures are read directly from the project's results file.</p>
        </div>
        <div style="margin-top:70px">{model_rows_html()}</div>
        <div class="btn-row"><a class="btn ghost" href="?page=models" target="_self">Model intelligence <span class="arr"></span></a></div>
        </div>
        """
    )

    # ---------------- D · sensor reliability ----------------
    cols = ["Clean"] + [f"{lvl}% noise" for lvl in NOISE_LEVELS if lvl > 0] + ["One sensor missing"]
    head = '<div class="xh">Accuracy</div>' + "".join(f'<div class="xh">{esc(c)}</div>' for c in cols)
    grid = [head]
    for model in MODEL_NAMES:
        clean_val = float(_clean.loc[_clean["Model"] == model, "Accuracy"].iloc[0])
        cells = [f'<div class="xm">{esc(model)}</div>', f'<div class="xv"><b>{pct(clean_val)}%</b><small>Baseline</small></div>']
        for lvl in [l for l in NOISE_LEVELS if l > 0]:
            val = float(_noise[(_noise["Model"] == model) & (_noise["Level"] == lvl)]["Accuracy"].iloc[0])
            cells.append(f'<div class="xv"><b>{pct(val)}%</b><small>{lose((clean_val - val) * 100)} pts</small></div>')
        sub = _missing_df[_missing_df["Model"] == model]
        worst = sub.loc[sub["Accuracy"].idxmin()]
        cells.append(
            f'<div class="xv worst"><b>{pct(worst["Accuracy"])}%</b>'
            f'<small>Worst case: {esc(worst["Sensor"])} &nbsp;{lose((clean_val - worst["Accuracy"]) * 100)} pts</small></div>'
        )
        grid.append("".join(cells))
    md(
        f"""
        <div class="bleed sec" id="sensor-reliability" style="padding-bottom:30px">
        <div class="bgfill img-reliability"></div>
        <div class="sec-in">
        <div class="reveal" style="max-width:1000px">
        <div class="eyebrow">04 &nbsp;/&nbsp; Sensor reliability</div>
        <div class="display md">When the signal <em>degrades.</em></div>
        <div class="rule"></div>
        <p class="lede">Real sensors drift, add noise and fail. Each model was stress-tested with simulated Gaussian noise at three severities and with every sensor removed in turn. These are controlled experiments, not physical failure tests.</p>
        </div>
        <div class="xwrap reveal"><div class="xgrid">{"".join(grid)}</div></div>
        </div>
        </div>
        """
    )
    with st.container():
        fig = go.Figure()
        for model in MODEL_NAMES:
            sub = NOISE_ALL[NOISE_ALL["Model"] == model].sort_values("Level")
            fig.add_trace(
                go.Scatter(
                    x=[level_label(l) for l in sub["Level"]],
                    y=sub["Accuracy"] * 100,
                    mode="lines+markers",
                    name=model,
                    line=dict(width=3, color=MODEL_COLORS.get(model, GOLD)),
                    marker=dict(size=10, color="#0a0a09", line=dict(width=2, color=MODEL_COLORS.get(model, GOLD))),
                    hovertemplate="%{x}<br>%{y:.2f}% accuracy<extra>" + model + "</extra>",
                )
            )
        fig.update_yaxes(title="Accuracy (%)")
        show_chart(luxe(fig, 380), "home_noise_chart")
    md(
        """
        <div class="reveal" style="padding:10px 0 0"><div class="btn-row" style="margin-top:20px">
        <a class="btn gold" href="?page=reliability" target="_self">Enter the reliability lab <span class="arr"></span></a></div></div>
        """
    )

    # ---------------- E · research philosophy ----------------
    md(
        """
        <div class="bleed sec" style="padding-bottom:0">
        <div class="reveal" style="max-width:1250px">
        <div class="eyebrow">05 &nbsp;/&nbsp; Research philosophy</div>
        <div class="serif" style="font-size:clamp(2rem,4.6vw,4.6rem)">It is not enough to ask whether a model can identify a fault. We ask how <em style="color:#D9BC7E">dependable</em> that prediction remains when the sensors become unreliable.</div>
        </div>
        <div class="pillars reveal">
        <div><div class="pn">I</div><div class="pt">Identify</div><p class="small">Can a model recognise the engine condition from clean sensor readings?</p></div>
        <div><div class="pn">II</div><div class="pt">Stress</div><p class="small">What happens to that prediction as readings become noisy or a sensor disappears?</p></div>
        <div><div class="pn">III</div><div class="pt">Understand</div><p class="small">Which sensors does each model depend on most, and how fragile is that dependence?</p></div>
        </div>
        <div class="btn-row"><a class="btn ghost" href="?page=research" target="_self">Read the research <span class="arr"></span></a></div>
        </div>
        """
    )


# =========================================================
# PAGE: ENGINE DIAGNOSTICS
# =========================================================


def run_model(model, frame: pd.DataFrame):
    prediction = int(model.predict(frame)[0])
    classes = [int(c) for c in getattr(model, "classes_", sorted(FAULT_LABELS))]
    scores = None
    kind = None
    if hasattr(model, "predict_proba"):
        try:
            scores = np.asarray(model.predict_proba(frame))[0] * 100
            kind = "probability output"
        except Exception:
            scores = None
    if scores is None and hasattr(model, "decision_function"):
        try:
            raw = np.asarray(model.decision_function(frame)).reshape(-1)
            if raw.shape[0] == len(classes):
                exp = np.exp(raw - raw.max())
                scores = exp / exp.sum() * 100
                kind = "normalised decision score"
        except Exception:
            scores = None
    return prediction, classes, scores, kind


def grouped_features() -> list[tuple[str, list[str]]]:
    used: set[str] = set()
    groups = []
    for title, names in SENSOR_GROUPS:
        present = [n for n in names if n in features]
        used.update(present)
        if present:
            groups.append((title, present))
    leftover = [f for f in features if f not in used]
    if leftover:
        groups.append(("Additional channels", leftover))
    return groups


def page_diagnostics() -> None:
    page_hero(
        "engine",
        "Engine diagnostics",
        "Read the <em>engine.</em>",
        "Select a model, enter the sensor measurements and run the trained classifier. "
        "The prediction is returned as a diagnostic report with the model's score for every condition.",
    )
    spacer()

    medians = df[features].median()
    lows, highs = df[features].min(), df[features].max()

    left, right = st.columns([1.25, 1], gap="large")

    with left:
        md('<div class="eyebrow">Sensor telemetry input</div>')
        with st.form("diagnosis_form"):
            selected_model = st.radio(
                "Diagnostic model",
                MODEL_NAMES,
                index=MODEL_NAMES.index("Random Forest") if "Random Forest" in MODEL_NAMES else 0,
                horizontal=True,
            )
            st.caption(
                "Initial values are dataset medians, provided for interface testing. "
                "Replace them with actual measurements."
            )
            input_values: dict[str, float] = {}
            groups = grouped_features()
            sub_cols = st.columns(len(groups) if len(groups) <= 3 else 3, gap="medium")
            for g_i, (title, names) in enumerate(groups):
                with sub_cols[g_i % len(sub_cols)]:
                    md(f'<div class="fgroup">{esc(title)}</div>')
                    for name in names:
                        input_values[name] = st.number_input(
                            name,
                            value=float(medians[name]),
                            format="%.4f",
                            help=f"Dataset range: {lows[name]:.4f} to {highs[name]:.4f}",
                            key=f"sensor_{name}",
                        )
            submitted = st.form_submit_button("Run engine diagnostics")

    with right:
        md('<div class="eyebrow">Diagnostic output</div>')
        if not submitted:
            md(
                """
                <div class="idle"><div class="rl">Awaiting telemetry</div>
                <div class="rr">Run the diagnostics to generate a report</div>
                <p>The report shows the predicted engine condition, the model's output score for each of the four conditions and a cross-check against the other two models.</p></div>
                """
            )
        else:
            frame = pd.DataFrame([input_values], columns=features)
            model = models[selected_model]
            prediction, classes, scores, kind = run_model(model, frame)
            fault_name = FAULT_LABELS.get(prediction, "Unknown condition")
            stamp = datetime.now(timezone.utc).strftime("%d %b %Y &middot; %H:%M UTC")

            top_text, margin_text = "n/a", "n/a"
            if scores is not None:
                order = np.argsort(scores)[::-1]
                top_text = f"{scores[order[0]]:.1f}%"
                if len(order) > 1:
                    margin_text = f"{scores[order[0]] - scores[order[1]]:.1f} pts"
            md(
                f"""
                <div class="report">
                <div class="rh"><b>Diagnostic report</b><span>{stamp}</span></div>
                <div class="rl">Predicted engine condition</div>
                <div class="rr">{esc(fault_name)}</div>
                <div class="rm">
                <div><small>Fault code</small><span>{prediction}</span></div>
                <div><small>Model</small><span style="font-size:1.05rem;line-height:1.5">{esc(selected_model)}</span></div>
                <div><small>Top score</small><span>{top_text}</span></div>
                </div>
                <p class="rd">{esc(FAULT_NOTES.get(prediction, ""))} Margin over the runner-up: {margin_text}.</p>
                </div>
                """
            )

            if scores is not None:
                labels = [FAULT_LABELS.get(c, str(c)) for c in classes]
                colors = [GOLD if c == prediction else "#4a4030" for c in classes]
                fig = go.Figure(
                    go.Bar(
                        x=scores,
                        y=labels,
                        orientation="h",
                        marker=dict(color=colors, line=dict(color="rgba(217,188,126,.5)", width=1)),
                        text=[f"{s:.1f}%" for s in scores],
                        textposition="outside",
                        cliponaxis=False,
                        hovertemplate="%{y}<br>%{x:.2f}%<extra></extra>",
                    )
                )
                fig.update_xaxes(range=[0, max(100, float(scores.max()) * 1.18)], title=f"Model score (%), {kind}")
                fig.update_yaxes(autorange="reversed")
                show_chart(luxe(fig, 300, legend=False, margin=dict(l=10, r=40, t=14, b=10)), "diag_scores")
                st.caption(
                    "Scores are raw model outputs for this single input. They are not calibrated "
                    "probabilities of a real-world engine fault."
                )

    if submitted:
        spacer()
        md('<div class="eyebrow">Cross-model check</div>')
        rows, hl = [], {}
        for r_i, (name, mdl) in enumerate(models.items()):
            p, cls_, sc, _ = run_model(mdl, frame)
            top = f"{np.max(sc):.1f}%" if sc is not None else "n/a"
            rows.append([name, f"{p} · {FAULT_LABELS.get(p, 'Unknown')}", top])
            if name == selected_model:
                hl[(r_i, 0)] = "hl"
        md(html_table(["Model", "Predicted condition", "Top score"], rows, {2}, hl))
        spacer()
        md(
            """
            <div class="notice"><b>Experimental prediction.</b> This output is produced by machine-learning models trained on a
            laboratory dataset. It is not a certified mechanical diagnosis and must not replace inspection by a qualified technician.</div>
            """
        )
    else:
        spacer(small=True)
        md(
            """
            <div class="notice"><b>Experimental system.</b> Predictions are produced by models trained on a laboratory dataset and are
            not a certified mechanical diagnosis.</div>
            """
        )


# =========================================================
# PAGE: MODEL INTELLIGENCE
# =========================================================


def class_frame() -> pd.DataFrame:
    rows = []
    for model, values in metrics.items():
        report = values["classification_report"]
        for key in class_keys(report):
            code = int(key)
            rows.append(
                {
                    "Model": model,
                    "Class": code,
                    "Fault": FAULT_LABELS.get(code, f"Class {code}"),
                    "Precision": report[key]["precision"],
                    "Recall": report[key]["recall"],
                    "F1-score": report[key]["f1-score"],
                    "Support": report[key]["support"],
                }
            )
    return pd.DataFrame(rows)


def page_models() -> None:
    page_hero(
        "intelligence",
        "Model intelligence",
        "Three models.<br>One <em>benchmark.</em>",
        f"All models were evaluated on the same stratified, held-out test set of {N_TEST:,} records. "
        "Every figure on this page is read live from the project's results file.",
    )
    spacer()
    best = max(metrics, key=lambda m: metrics[m]["accuracy"])
    best_f1 = max(metrics, key=lambda m: metrics[m]["macro_f1"])
    md(
        f"""
        <div class="stats reveal" style="--cols:4">
        <div><div class="sv">{pct(metrics[best]["accuracy"])}<sup>%</sup></div><div class="sl">Best accuracy</div></div>
        <div><div class="sv" style="font-size:clamp(1.5rem,2.4vw,2.3rem);padding-top:10px">{esc(best)}</div><div class="sl">Leading model</div></div>
        <div><div class="sv">{pct(metrics[best_f1]["macro_f1"])}<sup>%</sup></div><div class="sl">Best macro F1</div></div>
        <div><div class="sv">{N_TEST:,}</div><div class="sl">Test records</div></div>
        </div>
        """
    )
    md(f'<div class="spacer"></div><div>{model_rows_html()}</div><div class="spacer"></div>')

    tab_overall, tab_class = st.tabs(["Overall performance", "Class-wise analysis"])

    with tab_overall:
        spacer(small=True)
        c1, c2 = st.columns(2, gap="large")
        for col, metric_key, title, k in [
            (c1, "accuracy", "Accuracy", "acc"),
            (c2, "macro_f1", "Macro F1", "f1"),
        ]:
            with col:
                md(f'<div class="eyebrow">{title} comparison</div>')
                vals = [metrics[m][metric_key] * 100 for m in metrics]
                fig = go.Figure(
                    go.Bar(
                        x=list(metrics.keys()),
                        y=vals,
                        marker=dict(
                            color=[MODEL_COLORS.get(m, GOLD) for m in metrics],
                            line=dict(color="rgba(243,237,224,.5)", width=1),
                        ),
                        text=[f"{v:.2f}%" for v in vals],
                        textposition="outside",
                        cliponaxis=False,
                        hovertemplate="%{x}<br>%{y:.2f}%<extra></extra>",
                    )
                )
                fig.update_yaxes(range=[0, 100], title=f"{title} (%)")
                show_chart(luxe(fig, 400, legend=False, margin=dict(l=10, r=10, t=30, b=10)), f"models_overall_{k}")
        md('<div class="eyebrow" style="margin-top:20px">Summary table</div>')
        rows = [[m, f"{pct(v['accuracy'])}%", f"{pct(v['macro_f1'])}%"] for m, v in metrics.items()]
        md(html_table(["Model", "Accuracy", "Macro F1"], rows, {1, 2}, {(i, 0): ("hl" if m == best else "") for i, m in enumerate(metrics)}))

    with tab_class:
        spacer(small=True)
        cf = class_frame()
        c1, c2 = st.columns([1, 2], gap="large")
        with c1:
            metric_choice = st.radio("Metric", ["Precision", "Recall", "F1-score"], index=2, horizontal=True, key="cw_metric")
        with c2:
            model_choice = st.selectbox("Model for detail view", MODEL_NAMES, index=MODEL_NAMES.index(best), key="cw_model")

        md(f'<div class="eyebrow" style="margin-top:14px">{esc(metric_choice)} by fault class</div>')
        fig = go.Figure()
        for model in metrics:
            sub = cf[cf["Model"] == model].sort_values("Class")
            fig.add_trace(
                go.Bar(
                    name=model,
                    x=sub["Fault"],
                    y=sub[metric_choice] * 100,
                    marker=dict(color=MODEL_COLORS.get(model, GOLD), line=dict(color="rgba(243,237,224,.35)", width=1)),
                    text=[f"{v * 100:.1f}" for v in sub[metric_choice]],
                    textposition="outside",
                    cliponaxis=False,
                    hovertemplate="%{x}<br>" + model + ": %{y:.2f}%<extra></extra>",
                )
            )
        fig.update_layout(barmode="group", bargap=0.28)
        fig.update_yaxes(range=[0, 112], title=f"{metric_choice} (%)")
        show_chart(luxe(fig, 430), "models_class_bars")

        md(f'<div class="eyebrow" style="margin-top:20px">{esc(model_choice)} &nbsp;/&nbsp; precision, recall and F1 per class</div>')
        sub = cf[cf["Model"] == model_choice].sort_values("Class")
        z = sub[["Precision", "Recall", "F1-score"]].to_numpy() * 100
        fig = go.Figure(
            go.Heatmap(
                z=z,
                x=["Precision", "Recall", "F1-score"],
                y=list(sub["Fault"]),
                text=[[f"{v:.1f}%" for v in row] for row in z],
                texttemplate="%{text}",
                textfont=dict(size=14, color=IVORY, family="Jost"),
                colorscale=[[0, "#14120d"], [0.5, "#4d3d22"], [1, "#A4854A"]],
                zmin=0,
                zmax=100,
                showscale=False,
                xgap=3,
                ygap=3,
                hovertemplate="%{y}<br>%{x}: %{z:.2f}%<extra></extra>",
            )
        )
        fig.update_yaxes(autorange="reversed")
        fig.update_xaxes(side="top")
        show_chart(luxe(fig, 330, legend=False, margin=dict(l=10, r=10, t=40, b=10)), "models_class_heat")

        weakest = sub.loc[sub["F1-score"].idxmin()]
        strongest = sub.loc[sub["F1-score"].idxmax()]
        md(
            f"""
            <p class="small" style="margin-top:10px">For {esc(model_choice)}, the strongest class by F1-score is
            <b style="color:#D9BC7E">{esc(strongest["Fault"])}</b> ({strongest["F1-score"] * 100:.2f}%) and the weakest is
            <b style="color:#D9BC7E">{esc(weakest["Fault"])}</b> ({weakest["F1-score"] * 100:.2f}%).</p>
            """
        )

        md('<div class="eyebrow" style="margin-top:36px">Complete class-wise table</div>')
        rows = [
            [r["Model"], r["Fault"], f"{r['Precision'] * 100:.2f}%", f"{r['Recall'] * 100:.2f}%",
             f"{r['F1-score'] * 100:.2f}%", f"{int(r['Support']):,}"]
            for _, r in cf.sort_values(["Model", "Class"]).iterrows()
        ]
        md(html_table(["Model", "Condition", "Precision", "Recall", "F1-score", "Support"], rows, {2, 3, 4, 5}))


# =========================================================
# PAGE: SENSOR RELIABILITY LAB
# =========================================================


def metric_series(frame: pd.DataFrame, model: str, column: str, key_col: str) -> pd.Series:
    sub = frame[frame["Model"] == model]
    return pd.Series(sub[column].to_numpy(), index=sub[key_col].to_numpy())


def page_reliability() -> None:
    page_hero(
        "reliability",
        "Sensor reliability lab",
        "When sensors <em>fail.</em>",
        "The central research question: how dependable does a prediction remain when sensor measurements become "
        "noisy or unavailable? Explore every result from the experiment, model by model.",
    )
    spacer()

    ctl1, ctl2, ctl3 = st.columns([1.55, 0.9, 0.95], gap="large")
    with ctl1:
        experiment = st.radio("Experiment", [EXPERIMENT_NOISE, EXPERIMENT_MISSING], horizontal=True, key="lab_experiment")
    with ctl2:
        model_choice = st.selectbox("Model", ["All models", *MODEL_NAMES], key="lab_model")
    with ctl3:
        metric_label = st.radio("Metric", ["Accuracy", "Macro F1"], horizontal=True, key="lab_metric")
    column = "Accuracy" if metric_label == "Accuracy" else "Macro_F1"
    selected_models = MODEL_NAMES if model_choice == "All models" else [model_choice]
    spacer()

    # ------------------------------------------------------------------
    if experiment == EXPERIMENT_NOISE:
        md(
            f"""
            <div class="reveal"><div class="eyebrow">Experiment 01 &nbsp;/&nbsp; Gaussian sensor noise</div>
            <div class="display sm">Clean baseline versus {", ".join(str(l) + "%" for l in NOISE_LEVELS if l > 0)} noise</div></div>
            """
        )
        spacer(small=True)
        cleans = {m: float(_clean.loc[_clean["Model"] == m, column].iloc[0]) for m in MODEL_NAMES}
        labels = [level_label(l) for l in NOISE_LEVELS]

        fig = go.Figure()
        linear = [m for m in ("PCA + LDA", "LDA Only") if m in selected_models]
        for model in selected_models:
            sub = NOISE_ALL[NOISE_ALL["Model"] == model].sort_values("Level")
            positions = []
            for _, row in sub.iterrows():
                if len(linear) == 2 and model in linear:
                    other = [m for m in linear if m != model][0]
                    other_val = float(NOISE_ALL[(NOISE_ALL["Model"] == other) & (NOISE_ALL["Level"] == row["Level"])][column].iloc[0])
                    positions.append("top center" if row[column] >= other_val else "bottom center")
                else:
                    positions.append("top center")
            fig.add_trace(
                go.Scatter(
                    x=[level_label(l) for l in sub["Level"]],
                    y=sub[column] * 100,
                    mode="lines+markers+text",
                    name=model,
                    text=[f"{v * 100:.2f}" for v in sub[column]],
                    textposition=positions,
                    textfont=dict(size=11, color=MODEL_COLORS.get(model, GOLD)),
                    line=dict(width=3.5, color=MODEL_COLORS.get(model, GOLD)),
                    marker=dict(size=11, color="#0a0a09", line=dict(width=2.5, color=MODEL_COLORS.get(model, GOLD))),
                    hovertemplate="%{x}<br>%{y:.2f}%<extra>" + model + "</extra>",
                )
            )
        fig.update_yaxes(title=f"{metric_label} (%)")
        fig.update_xaxes(categoryorder="array", categoryarray=labels)
        show_chart(luxe(fig, 470, margin=dict(l=10, r=10, t=40, b=10)), "lab_noise_lines")

        md('<div class="eyebrow" style="margin-top:26px">Performance lost to noise (percentage points)</div>')
        fig = go.Figure()
        for model in selected_models:
            sub = NOISE_ALL[(NOISE_ALL["Model"] == model) & (NOISE_ALL["Level"] > 0)].sort_values("Level")
            loss = (cleans[model] - sub[column]) * 100
            fig.add_trace(
                go.Bar(
                    name=model,
                    x=[level_label(l) for l in sub["Level"]],
                    y=loss,
                    marker=dict(color=MODEL_COLORS.get(model, GOLD), line=dict(color="rgba(243,237,224,.35)", width=1)),
                    text=[f"{v:.2f}" for v in loss],
                    textposition="outside",
                    cliponaxis=False,
                    hovertemplate="%{x}<br>%{y:.2f} points lost<extra>" + model + "</extra>",
                )
            )
        fig.update_layout(barmode="group", bargap=0.3)
        fig.update_yaxes(title=f"{metric_label} lost (points)", rangemode="tozero")
        show_chart(luxe(fig, 380, margin=dict(l=10, r=10, t=40, b=10)), "lab_noise_loss")

        md('<div class="eyebrow" style="margin-top:26px">Results table</div>')
        rows = []
        for model in selected_models:
            sub = NOISE_ALL[NOISE_ALL["Model"] == model].sort_values("Level")
            for _, r in sub.iterrows():
                rows.append([
                    model,
                    level_label(int(r["Level"])),
                    f"{r[column] * 100:.2f}%",
                    "baseline" if r["Level"] == 0 else f"{lose((cleans[model] - r[column]) * 100)} pts",
                ])
        md(html_table(["Model", "Condition", metric_label, "Change vs clean"], rows, {2, 3}))

    # ------------------------------------------------------------------
    else:
        md(
            f"""
            <div class="reveal"><div class="eyebrow">Experiment 02 &nbsp;/&nbsp; Individual missing sensor</div>
            <div class="display sm">Each of the {len(SENSORS)} sensors removed in turn</div></div>
            """
        )
        spacer(small=True)

        cleans = {m: float(_clean.loc[_clean["Model"] == m, column].iloc[0]) for m in MODEL_NAMES}
        values = {m: metric_series(_missing_df, m, column, "Sensor").reindex(SENSORS) for m in MODEL_NAMES}
        loss = {m: (cleans[m] - values[m]) * 100 for m in MODEL_NAMES}

        # headline findings, computed from the CSV
        finds = []
        for m in selected_models:
            top = loss[m].sort_values(ascending=False)
            finds.append(
                f'<div><div class="sv" style="font-size:clamp(1.4rem,2.2vw,2.1rem)">{esc(top.index[0])}</div>'
                f'<div class="sl">{esc(m)} · most damaging · {pts(top.iloc[0])} pts lost</div></div>'
            )
        md(f'<div class="stats reveal" style="--cols:{len(finds)}">{"".join(finds)}</div>')
        spacer(small=True)

        if model_choice == "All models":
            md(f'<div class="eyebrow">{esc(metric_label)} lost when each sensor is missing (points)</div>')
            z = np.array([[loss[m][s] for m in MODEL_NAMES] for s in SENSORS])
            order = np.argsort(-z.mean(axis=1))
            z = z[order]
            sensors_sorted = [SENSORS[i] for i in order]
            zmax = float(max(abs(z.min()), abs(z.max())))
            fig = go.Figure(
                go.Heatmap(
                    z=z,
                    x=MODEL_NAMES,
                    y=sensors_sorted,
                    zmid=0,
                    zmin=-zmax,
                    zmax=zmax,
                    colorscale=[[0, "#4F6877"], [0.5, "#14120d"], [1, "#A4854A"]],
                    text=[[f"{v:.2f}" for v in row] for row in z],
                    texttemplate="%{text}",
                    textfont=dict(size=12, color=IVORY),
                    xgap=3,
                    ygap=3,
                    colorbar=dict(title=dict(text="points lost", side="right"), thickness=10, outlinewidth=0),
                    hovertemplate="%{y} missing<br>%{x}: %{z:.2f} points lost<extra></extra>",
                )
            )
            fig.update_yaxes(autorange="reversed")
            fig.update_xaxes(side="top")
            show_chart(luxe(fig, 640, legend=False, margin=dict(l=10, r=10, t=44, b=10)), "lab_missing_heat")

            md('<div class="eyebrow" style="margin-top:26px">Side-by-side comparison</div>')
            fig = go.Figure()
            for m in MODEL_NAMES:
                fig.add_trace(
                    go.Bar(
                        name=m,
                        x=sensors_sorted,
                        y=[loss[m][s] for s in sensors_sorted],
                        marker=dict(color=MODEL_COLORS.get(m, GOLD), line=dict(color="rgba(243,237,224,.3)", width=1)),
                        hovertemplate="%{x} missing<br>%{y:.2f} points<extra>" + m + "</extra>",
                    )
                )
            fig.update_layout(barmode="group", bargap=0.25)
            fig.add_hline(y=0, line_color="rgba(243,237,224,.5)", line_width=1)
            fig.update_xaxes(tickangle=-35)
            fig.update_yaxes(title=f"{metric_label} lost (points)")
            show_chart(luxe(fig, 470), "lab_missing_group")

            md('<div class="eyebrow" style="margin-top:26px">Sensor ranking (most damaging first)</div>')
            ranks = {m: loss[m].rank(ascending=False, method="min").astype(int) for m in MODEL_NAMES}
            rows, cls = [], {}
            for r_i, s in enumerate(sensors_sorted):
                row = [s]
                for c_i, m in enumerate(MODEL_NAMES, start=1):
                    row.append(f"{pts(loss[m][s])}  (#{ranks[m][s]})")
                    if loss[m][s] < 0:
                        cls[(r_i, c_i)] = "gain"
                rows.append(row)
            md(html_table(["Sensor missing", *[f"{m} · loss (rank)" for m in MODEL_NAMES]], rows, set(range(1, len(MODEL_NAMES) + 1)), cls))

        else:
            m = model_choice
            order = loss[m].sort_values(ascending=False)
            sensors_sorted = list(order.index)
            clean_v = cleans[m] * 100

            c1, c2 = st.columns(2, gap="large")
            with c1:
                md(f'<div class="eyebrow">{esc(metric_label)} lost per missing sensor (points)</div>')
                fig = go.Figure(
                    go.Bar(
                        y=sensors_sorted,
                        x=order.values,
                        orientation="h",
                        marker=dict(
                            color=[GOLD if v >= 0 else SLATE for v in order.values],
                            line=dict(color="rgba(243,237,224,.3)", width=1),
                        ),
                        text=[pts(v) for v in order.values],
                        textposition="outside",
                        cliponaxis=False,
                        hovertemplate="%{y} missing<br>%{x:.2f} points lost<extra></extra>",
                    )
                )
                fig.update_yaxes(autorange="reversed")
                fig.update_xaxes(title="points lost (negative = slightly better without it)")
                show_chart(luxe(fig, 600, legend=False, margin=dict(l=10, r=50, t=20, b=10)), "lab_missing_rank")
            with c2:
                md(f'<div class="eyebrow">Clean versus missing-sensor {esc(metric_label.lower())}</div>')
                xs, ys = [], []
                for s in sensors_sorted:
                    xs += [clean_v, values[m][s] * 100, None]
                    ys += [s, s, None]
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", line=dict(color="rgba(217,188,126,.45)", width=2),
                                         hoverinfo="skip", showlegend=False))
                fig.add_trace(go.Scatter(x=[clean_v] * len(sensors_sorted), y=sensors_sorted, mode="markers", name="Clean",
                                         marker=dict(size=11, color="#0a0a09", line=dict(width=2, color=IVORY)),
                                         hovertemplate="Clean: %{x:.2f}%<extra></extra>"))
                fig.add_trace(go.Scatter(x=[values[m][s] * 100 for s in sensors_sorted], y=sensors_sorted, mode="markers",
                                         name="Sensor missing",
                                         marker=dict(size=11, color=GOLD, line=dict(width=1, color="#0a0a09")),
                                         hovertemplate="%{y} missing: %{x:.2f}%<extra></extra>"))
                fig.update_yaxes(autorange="reversed")
                fig.update_xaxes(title=f"{metric_label} (%)")
                show_chart(luxe(fig, 600, margin=dict(l=10, r=10, t=44, b=10)), "lab_missing_dumbbell")

            md('<div class="eyebrow" style="margin-top:20px">Sensor ranking</div>')
            rows, cls = [], {}
            for r_i, s in enumerate(sensors_sorted, start=1):
                rows.append([
                    r_i, s, f"{clean_v:.2f}%", f"{values[m][s] * 100:.2f}%", f"{pts(loss[m][s])} pts",
                ])
                if loss[m][s] < 0:
                    cls[(r_i - 1, 4)] = "gain"
                elif r_i <= 3:
                    cls[(r_i - 1, 4)] = "hl"
            md(html_table(["Rank", "Sensor missing", f"Clean {metric_label}", f"Missing {metric_label}", "Points lost"], rows, {0, 2, 3, 4}, cls))

        md(
            """
            <p class="small" style="margin-top:14px">Loss is the clean score minus the score with that sensor missing, in percentage points.
            A negative loss means the model scored slightly higher without that sensor; it does not mean the sensor is unimportant in general.</p>
            """
        )

    # ------------------------------------------------------------------
    spacer()
    md(
        """
        <div class="notice reveal"><b>How these experiments were run.</b> Noise was simulated with Gaussian perturbations.
        Missing sensors were simulated by replacing a sensor's readings with its training-set median. These are controlled
        software experiments on a laboratory dataset, not physical sensor-failure tests.</div>
        """
    )
    spacer(small=True)
    st.download_button(
        "Download full results (CSV)",
        data=reliability.to_csv(index=False),
        file_name="sensor_reliability_results.csv",
        mime="text/csv",
    )


# =========================================================
# PAGE: RESEARCH
# =========================================================


def rblock(num: str, title: str, body: str) -> str:
    return (
        f'<div class="rblock reveal"><div><div class="rn">{num}</div><div class="rt">{title}</div></div>'
        f'<div class="rb">{body}</div></div>'
    )


def page_research() -> None:
    page_hero(
        "philosophy",
        "Research",
        "Method, data and <em>limits.</em>",
        "How AutoSense AI was built, what it was tested on, and what it should not be used to claim.",
    )
    spacer()

    counts = df["Fault"].value_counts().sort_index()
    stats = (
        f'<div class="stats reveal" style="--cols:4">'
        f'<div><div class="sv">{N_RECORDS:,}</div><div class="sl">Records</div></div>'
        f'<div><div class="sv">{len(features)}</div><div class="sl">Sensor features</div></div>'
        f'<div><div class="sv">{df["Fault"].nunique()}</div><div class="sl">Fault categories</div></div>'
        f'<div><div class="sv">80:20</div><div class="sl">Stratified split</div></div></div>'
    )
    md(stats)
    spacer(small=True)

    fault_rows = "".join(
        f"<li><b style='color:#D9BC7E'>{code}</b> &nbsp; {esc(FAULT_LABELS.get(int(code), code))} "
        f"<span style='color:#9C927F'>&mdash; {n:,} records</span></li>"
        for code, n in counts.items()
    )

    blocks = [
        rblock("01", "Research objective",
               "<p>To investigate how engine-fault classification performance changes when sensor measurements become noisy or "
               "when individual sensor measurements become unavailable.</p>"
               "<p>AutoSense AI asks not only whether a model can identify a fault, but how dependable that prediction remains "
               "when the sensors it relies on become unreliable.</p>"),
        rblock("02", "Dataset",
               f"<p><b style='color:#D9BC7E;font-weight:500'>EngineFaultDB</b>, collected in controlled laboratory conditions "
               f"using a C14NE spark-ignition engine. After removing duplicate rows the working dataset contains "
               f"{N_RECORDS:,} records, {len(features)} sensor features and {df['Fault'].nunique()} fault categories.</p>"
               f"<ul>{fault_rows}</ul>"
               f"<p style='margin-top:18px'>Sensor features: {esc(', '.join(features))}.</p>"
               '<p>References: <a href="https://github.com/leoxthomas/EngineFaultDB" target="_blank" rel="noopener">EngineFaultDB repository</a> '
               'and <a href="https://ieeexplore.ieee.org/abstract/document/10311597" target="_blank" rel="noopener">the original IEEE paper</a>.</p>'),
        rblock("03", "PCA + LDA",
               "<p>Principal Component Analysis reduces the original sensor features before Linear Discriminant Analysis "
               "performs the classification.</p>"),
        rblock("04", "LDA only",
               "<p>Linear Discriminant Analysis classifies engine conditions using the original standardized sensor features, "
               "without a preceding dimensionality-reduction step.</p>"),
        rblock("05", "Random Forest",
               "<p>An ensemble of decision trees classifies engine faults from the sensor measurements. Each tree votes and the "
               "ensemble returns the final condition.</p>"),
        rblock("06", "Evaluation protocol",
               f"<ul><li>Stratified 80:20 train-test split: {N_TRAIN:,} training and {N_TEST:,} test records.</li>"
               "<li>All three models are evaluated on the same held-out test set.</li>"
               "<li>Accuracy and macro F1 are reported; class-wise precision, recall and F1 are available on the Model Intelligence page.</li></ul>"),
        rblock("07", "Sensor noise experiment",
               "<ul><li>Gaussian noise is added to the sensor readings at 5%, 10% and 20% of each feature's training-set standard deviation.</li>"
               "<li>Each noisy test set is compared with the clean baseline.</li>"
               "<li>The noise is simulated; it is not measured sensor noise.</li></ul>"),
        rblock("08", "Missing sensor experiment",
               f"<ul><li>Each of the {len(SENSORS)} sensors is removed in turn by replacing its readings with the training-set median.</li>"
               "<li>The loss in accuracy and macro F1, in percentage points, ranks how strongly each model depends on each sensor.</li>"
               "<li>These are controlled experiments, not physical sensor-failure tests.</li></ul>"),
    ]
    md("".join(blocks))

    md('<div class="eyebrow" style="margin-top:30px">Class balance of the working dataset</div>')
    fig = go.Figure(
        go.Bar(
            x=[FAULT_LABELS.get(int(c), str(c)) for c in counts.index],
            y=counts.values,
            marker=dict(color=[GOLD, "#C9AC6E", "#A98A52", BRONZE][: len(counts)], line=dict(color="rgba(243,237,224,.3)", width=1)),
            text=[f"{v:,}" for v in counts.values],
            textposition="outside",
            cliponaxis=False,
            hovertemplate="%{x}<br>%{y:,} records<extra></extra>",
        )
    )
    fig.update_yaxes(title="Records")
    show_chart(luxe(fig, 340, legend=False, margin=dict(l=10, r=10, t=24, b=10)), "research_balance")

    md(
        rblock("09", "Dataset limitations",
               "<ul><li>The data was collected in controlled laboratory conditions on a single C14NE spark-ignition engine.</li>"
               "<li>The models have not been validated for other engines, vehicles or real-world diagnostic environments.</li>"
               "<li>The noise and missing-sensor results are simulations and should not be read as physical sensor-failure behaviour.</li>"
               "<li>Predictions are experimental and are not a certified mechanical diagnosis.</li></ul>"
               )
    )
    spacer(small=True)
    md(
        """
        <div class="notice reveal"><b>Important.</b> AutoSense AI is a research prototype. Do not rely on it to make safety-critical
        decisions about a vehicle. Always consult a qualified technician.</div>
        """
    )


# =========================================================
# ROUTER
# =========================================================


def current_page() -> str:
    try:
        value = st.query_params.get("page", "home")
    except Exception:  # very old Streamlit
        value = st.experimental_get_query_params().get("page", ["home"])
    if isinstance(value, list):
        value = value[0] if value else "home"
    return value if value in PAGES else "home"


active_page = current_page()
render_nav(active_page)

if active_page == "home":
    page_home()
elif active_page == "diagnostics":
    page_diagnostics()
elif active_page == "models":
    page_models()
elif active_page == "reliability":
    page_reliability()
else:
    page_research()

render_footer()
