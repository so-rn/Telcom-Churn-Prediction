"""
Visual theme for the churn app: Dark Premium + Glassmorphism.

Everything that makes the app look like a top-tier fintech dashboard lives
here - the color tokens, the injected CSS, and the Plotly template. Pages
import COLORS / plotly_layout() and call inject_css() once at startup.
"""

from __future__ import annotations

import re

import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st
import streamlit.components.v1 as components

# ---------------------------------------------------------------------------
# Design tokens
# ---------------------------------------------------------------------------
COLORS = {
    "bg":        "#0B0E14",
    "bg_soft":   "#11151F",
    "card":      "rgba(255, 255, 255, 0.045)",
    "card_brd":  "rgba(255, 255, 255, 0.09)",
    "text":      "#F5F7FA",
    "muted":     "#8A93A5",
    "cyan":      "#00E5FF",
    "violet":    "#7C5CFC",
    "green":     "#2ED47A",
    "amber":     "#FFB547",
    "orange":    "#FF8A4C",
    "red":       "#FF5C7C",
}

# Risk-tier color mapping (shared across pages)
TIER_COLORS = {
    "Low":      COLORS["green"],
    "Medium":   COLORS["amber"],
    "High":     COLORS["orange"],
    "Critical": COLORS["red"],
}

CHART_SEQUENCE = [
    COLORS["cyan"], COLORS["violet"], COLORS["green"],
    COLORS["amber"], COLORS["red"], COLORS["orange"],
]


# ---------------------------------------------------------------------------
# Plotly template
# ---------------------------------------------------------------------------
def register_plotly_template() -> None:
    """Register and activate a dark template matching the app theme."""
    tmpl = go.layout.Template()
    tmpl.layout = go.Layout(
        font=dict(family="Inter, sans-serif", color=COLORS["text"], size=13),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        colorway=CHART_SEQUENCE,
        margin=dict(l=10, r=10, t=40, b=10),
        xaxis=dict(gridcolor="rgba(255,255,255,0.06)", zerolinecolor="rgba(255,255,255,0.10)",
                   linecolor="rgba(255,255,255,0.12)", tickcolor="rgba(255,255,255,0.12)"),
        yaxis=dict(gridcolor="rgba(255,255,255,0.06)", zerolinecolor="rgba(255,255,255,0.10)",
                   linecolor="rgba(255,255,255,0.12)", tickcolor="rgba(255,255,255,0.12)"),
        legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor="rgba(255,255,255,0.08)"),
        hoverlabel=dict(bgcolor=COLORS["bg_soft"], bordercolor=COLORS["card_brd"],
                        font=dict(family="Inter, sans-serif", color=COLORS["text"])),
    )
    pio.templates["churn_dark"] = tmpl
    pio.templates.default = "churn_dark"


def style_fig(fig: go.Figure, height: int = 320) -> go.Figure:
    """Apply consistent height/template to a Plotly figure.

    Force an empty title string: with Plotly 6 + Streamlit a figure with no
    title text otherwise renders the literal word "undefined" in the SVG.
    """
    fig.update_layout(template="churn_dark", height=height, title_text="")
    return fig


# ---------------------------------------------------------------------------
# CSS
# ---------------------------------------------------------------------------
_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');
@import url('https://fonts.googleapis.com/css2?family=Material+Symbols+Rounded:opsz,wght,FILL,GRAD@20..48,100..700,0..1,-50..200');

.material-symbols-rounded {
  font-family: 'Material Symbols Rounded';
  font-weight: normal; font-style: normal; line-height: 1;
  letter-spacing: normal; text-transform: none; display: inline-block;
  white-space: nowrap; direction: ltr;
  font-variation-settings: 'FILL' 1, 'wght' 500, 'GRAD' 0, 'opsz' 48;
}

:root {
  --bg: #0B0E14;
  --cyan: #00E5FF;
  --violet: #7C5CFC;
  --text: #F5F7FA;
  --muted: #8A93A5;
  --card: rgba(255,255,255,0.045);
  --card-brd: rgba(255,255,255,0.09);
}

/* Base */
.stApp {
  background:
    radial-gradient(1100px 600px at 12% -8%, rgba(124,92,252,0.16), transparent 60%),
    radial-gradient(1000px 560px at 92% 6%, rgba(0,229,255,0.12), transparent 55%),
    #0B0E14;
  color: var(--text);
  font-family: 'Inter', sans-serif;
}
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

/* Trim default chrome, but KEEP the sidebar expand button (it lives in the
   header toolbar — hiding the whole header/toolbar made a collapsed sidebar
   impossible to reopen). Make the header transparent and hide only the menu,
   deploy button and status widget. */
#MainMenu, footer { visibility: hidden; }
header[data-testid="stHeader"] { background: transparent; box-shadow: none; }
[data-testid="stMainMenu"],
[data-testid="stAppDeployButton"],
[data-testid="stToolbarActions"],
[data-testid="stStatusWidget"] { display: none !important; }
[data-testid="stExpandSidebarButton"] { visibility: visible !important; }
.block-container { padding-top: 2.2rem; padding-bottom: 3rem; max-width: 1280px; }

/* Headings */
h1, h2, h3 { font-family: 'Space Grotesk', sans-serif; letter-spacing: -0.5px; color: var(--text); }
.grad-text {
  background: linear-gradient(92deg, var(--cyan), var(--violet));
  -webkit-background-clip: text; -webkit-text-fill-color: transparent;
  background-clip: text;
}

/* Hero header */
.hero-title { font-size: 2.5rem; font-weight: 700; margin: 0; line-height: 1.1; }
.hero-sub { color: var(--muted); font-size: 1.02rem; margin-top: .35rem; font-weight: 400; }
.eyebrow {
  display:inline-block; font-size:.72rem; letter-spacing:2.5px; text-transform:uppercase;
  color: var(--cyan); font-weight:600; margin-bottom:.5rem;
  padding:.28rem .7rem; border:1px solid rgba(0,229,255,0.25); border-radius:999px;
  background: rgba(0,229,255,0.06);
}

/* Cinematic fade-in on load */
@keyframes fadeInUp { from { opacity: 0; transform: translateY(10px); }
                       to   { opacity: 1; transform: translateY(0);   } }
[data-testid="stPlotlyChart"], [data-testid="stDataFrame"] { animation: fadeInUp .6s ease both; }

/* Glass cards */
.glass {
  background: var(--card);
  backdrop-filter: blur(14px); -webkit-backdrop-filter: blur(14px);
  border: 1px solid var(--card-brd);
  border-radius: 18px; padding: 1.25rem 1.4rem;
  box-shadow: 0 8px 30px rgba(0,0,0,0.35);
  transition: transform .18s ease, box-shadow .18s ease, border-color .18s ease;
  animation: fadeInUp .5s ease both;
}
.glass:hover {
  transform: translateY(-2px);
  border-color: rgba(0,229,255,0.28);
  box-shadow: 0 14px 40px rgba(0,0,0,0.48);
}

/* KPI metric card */
.kpi { position: relative; overflow: hidden; min-height: 134px;
  display: flex; flex-direction: column; }
.kpi .label { color: var(--muted); font-size:.8rem; font-weight:500; letter-spacing:.4px; text-transform:uppercase; }
.kpi .value { font-family:'Space Grotesk',sans-serif; font-size:clamp(1.45rem, 2.1vw, 2.05rem); font-weight:700; margin-top:.3rem; line-height:1; white-space:nowrap; }
.kpi .delta { font-size:.82rem; margin-top:.45rem; font-weight:500; }
.kpi .spark { position:absolute; right:12px; top:16px; opacity:.16; font-size:1.9rem; }
.kpi .spark.material-symbols-rounded { font-variation-settings: 'FILL' 0, 'wght' 400, 'GRAD' 0, 'opsz' 40; }
.kpi.cyan   .value, .kpi.cyan   .spark { color: var(--cyan); }
.kpi.violet .value, .kpi.violet .spark { color: var(--violet); }
.kpi.green  .value, .kpi.green  .spark { color: #2ED47A; }
.kpi.red    .value, .kpi.red    .spark { color: #FF5C7C; }
.kpi.amber  .value, .kpi.amber  .spark { color: #FFB547; }

/* Pills / badges */
.pill { display:inline-block; padding:.3rem .8rem; border-radius:999px; font-size:.78rem; font-weight:600; }
.tier-badge { display:inline-flex; align-items:center; gap:.5rem; padding:.55rem 1.1rem; border-radius:14px;
  font-family:'Space Grotesk',sans-serif; font-size:1.15rem; font-weight:700; }

/* Section label */
.section { font-family:'Space Grotesk',sans-serif; font-size:1.15rem; font-weight:600; margin:.2rem 0 .1rem; }
.divider { height:1px; background:linear-gradient(90deg, transparent, rgba(255,255,255,0.12), transparent); margin:1.4rem 0; border:0; }

/* Sidebar */
[data-testid="stSidebar"] {
  background: linear-gradient(180deg, #0E1119, #0B0E14);
  border-right: 1px solid rgba(255,255,255,0.06);
}
[data-testid="stSidebar"] .stRadio label { color: var(--text); }

/* Buttons */
.stButton > button {
  background: linear-gradient(92deg, var(--cyan), var(--violet));
  color: #06070A; font-weight: 700; border: 0; border-radius: 12px;
  padding: .6rem 1.2rem; font-family:'Inter',sans-serif; transition: all .18s ease;
  box-shadow: 0 6px 18px rgba(0,229,255,0.18);
}
.stButton > button:hover { transform: translateY(-1px); box-shadow: 0 10px 26px rgba(124,92,252,0.32); color:#06070A; }

/* Inputs */
[data-baseweb="select"] > div, .stNumberInput input, .stTextInput input {
  background: rgba(255,255,255,0.04) !important; border-color: var(--card-brd) !important; color: var(--text) !important;
}
label, .stSlider label, .stSelectbox label { color: var(--muted) !important; font-weight:500; }

/* Tabs */
.stTabs [data-baseweb="tab-list"] { gap: 6px; }
.stTabs [data-baseweb="tab"] {
  background: rgba(255,255,255,0.03); border:1px solid var(--card-brd); border-radius:10px;
  padding: 8px 16px; color: var(--muted); font-weight:500;
}
.stTabs [aria-selected="true"] { background: rgba(0,229,255,0.10); color: var(--cyan); border-color: rgba(0,229,255,0.3); }

/* Dataframe */
[data-testid="stDataFrame"] { border:1px solid var(--card-brd); border-radius:14px; }

/* Custom watchlist table */
.wl-scroll { max-height: 470px; overflow-y: auto; border-radius: 10px; }
.wl { width:100%; border-collapse:separate; border-spacing:0; font-size:.88rem; }
.wl thead th { text-align:left; color:var(--muted); font-size:.68rem; text-transform:uppercase;
  letter-spacing:.9px; font-weight:600; padding:.7rem .6rem; border-bottom:1px solid var(--card-brd);
  position: sticky; top: 0; background:#10141E; z-index:2; }
.wl tbody td { padding:.72rem .6rem; border-bottom:1px solid rgba(255,255,255,.05); color:var(--text); }
.wl tbody tr:nth-child(even) td { background:rgba(255,255,255,.022); }
.wl tbody tr:last-child td { border-bottom:0; }
.wl tbody tr { transition: background .14s ease; }
.wl tbody tr:hover td { background:rgba(0,229,255,.07); }
.wl .wl-num { text-align:right; font-variant-numeric:tabular-nums; }
.wl .risk-wrap { display:flex; align-items:center; gap:.6rem; min-width:130px; }
.wl .risk-bar { flex:1; height:7px; border-radius:999px; background:rgba(255,255,255,.08); overflow:hidden; }
.wl .risk-fill { height:100%; border-radius:999px; }
.wl .risk-val { font-variant-numeric:tabular-nums; font-weight:700; min-width:40px; }
.wl-badge { display:inline-block; padding:.2rem .65rem; border-radius:999px; font-size:.72rem; font-weight:600; }

/* Scrollbar */
::-webkit-scrollbar { width: 9px; height: 9px; }
::-webkit-scrollbar-thumb { background: rgba(255,255,255,0.12); border-radius: 8px; }
::-webkit-scrollbar-track { background: transparent; }

/* ===== Sidebar: brand, nav, chips, credit ===== */
[data-testid="stSidebarNav"] { display: none; }   /* hide default nav; we render our own */
/* keep the sidebar a clean fixed width when open (Streamlit can let it balloon,
   which squeezes the main content); collapse still works since width shrinks below the cap */
[data-testid="stSidebar"] { max-width: 300px !important; }
[data-testid="stSidebar"][aria-expanded="true"] { width: 300px !important; min-width: 300px !important; }
/* Keep the sidebar permanently open (branding + nav stay visible; avoids the
   empty gap a collapsed sidebar leaves). Hide the collapse control so it can't
   be closed; the expand button is kept as a safety net for tiny screens that
   auto-collapse. */
[data-testid="stSidebarCollapseButton"] { display: none !important; }
/* naturally-flowing sidebar with generous breathing room */
[data-testid="stSidebar"] [data-testid="stVerticalBlock"] { gap: .5rem; }
[data-testid="stSidebar"] .block-container { padding-top: 1.6rem; }

.brand { display:flex; align-items:center; gap:.85rem; padding:.3rem .1rem .7rem;
  animation: brandIn .7s cubic-bezier(.2,.7,.2,1) both; }
.brand-logo {
  width:52px; height:52px; border-radius:16px; display:grid; place-items:center;
  font-size:1.65rem; color:#06070A; flex:0 0 auto;
  background:linear-gradient(135deg, var(--cyan), var(--violet));
  box-shadow:0 8px 24px rgba(0,229,255,.32);
  animation: logoPulse 3.4s ease-in-out infinite;
}
.brand-mono {
  font-family:'Space Grotesk',sans-serif; font-weight:700; font-size:1.4rem;
  letter-spacing:-1.5px; color:#06070A;
}
.brand-title {
  font-family:'Space Grotesk',sans-serif; font-weight:700; font-size:1.7rem; line-height:1.0;
  letter-spacing:-.6px;
  background:linear-gradient(92deg, var(--cyan), var(--violet), #38bdf8, var(--cyan));
  background-size:240% auto;
  -webkit-background-clip:text; -webkit-text-fill-color:transparent; background-clip:text;
  animation: shine 6.5s linear infinite;
}
.brand-sub { color:var(--muted); font-size:.7rem; letter-spacing:1.6px; margin-top:.42rem;
  text-transform:uppercase; font-weight:500; }

@keyframes shine { to { background-position: 240% center; } }
@keyframes logoPulse {
  0%,100% { box-shadow:0 8px 22px rgba(0,229,255,.28); transform:translateY(0) rotate(0deg); }
  50%     { box-shadow:0 12px 34px rgba(124,92,252,.55); transform:translateY(-2px) rotate(-1.5deg); }
}
@keyframes brandIn { from { opacity:0; transform:translateY(-10px); } to { opacity:1; transform:translateY(0); } }

.side-label { color:var(--muted); font-size:.64rem; letter-spacing:2.4px; text-transform:uppercase;
  margin:.9rem .15rem .75rem; font-weight:600; display:flex; align-items:center; gap:.55rem; }
.side-label::before { content:''; width:16px; height:2px; border-radius:2px; flex:0 0 auto;
  background:linear-gradient(90deg, var(--cyan), var(--violet)); }

/* custom page_link nav items */
[data-testid="stSidebarUserContent"] a[data-testid="stPageLink-NavLink"] {
  border-radius:13px; padding:.72rem .95rem; margin:7px 0; border:1px solid transparent;
  transition: background .18s ease, border-color .18s ease, transform .18s ease, box-shadow .18s ease;
}
[data-testid="stSidebarUserContent"] a[data-testid="stPageLink-NavLink"]:hover {
  background:rgba(255,255,255,.05); border-color:var(--card-brd); transform:translateX(4px);
}
[data-testid="stSidebarUserContent"] a[data-testid="stPageLink-NavLink"]:hover p { color:var(--text); }
[data-testid="stSidebarUserContent"] a[data-testid="stPageLink-NavLink"][aria-current="page"] {
  background:linear-gradient(92deg, rgba(0,229,255,.18), rgba(124,92,252,.14));
  border-color:rgba(0,229,255,.32);
  box-shadow: inset 3px 0 0 var(--cyan), 0 6px 18px rgba(0,229,255,.10);
}
[data-testid="stSidebarUserContent"] a[data-testid="stPageLink-NavLink"][aria-current="page"] p {
  color:var(--cyan) !important; font-weight:600;
}
/* nav labels + icon-as-tile */
[data-testid="stSidebarUserContent"] a[data-testid="stPageLink-NavLink"] { display:flex; align-items:center; gap:1.25rem; }
[data-testid="stSidebarUserContent"] a[data-testid="stPageLink-NavLink"] p { font-size:.92rem; font-weight:500; color:var(--muted); }
[data-testid="stSidebarUserContent"] a[data-testid="stPageLink-NavLink"] [data-testid="stIconMaterial"] {
  display:grid; place-items:center; width:36px; height:36px; border-radius:11px; flex:0 0 auto;
  background:rgba(255,255,255,.045); border:1px solid var(--card-brd);
  color:var(--muted); font-size:20px;
  transition: background .18s ease, border-color .18s ease, color .18s ease, box-shadow .18s ease, transform .18s ease;
}
[data-testid="stSidebarUserContent"] a[data-testid="stPageLink-NavLink"]:hover [data-testid="stIconMaterial"] {
  color:var(--cyan); border-color:rgba(0,229,255,.35); background:rgba(0,229,255,.09); transform:scale(1.05);
}
[data-testid="stSidebarUserContent"] a[data-testid="stPageLink-NavLink"][aria-current="page"] [data-testid="stIconMaterial"] {
  background:linear-gradient(135deg, var(--cyan), var(--violet)); border-color:transparent;
  color:#06070A; box-shadow:0 5px 16px rgba(0,229,255,.35);
}
/* Streamlit (>=1.58) marks the ACTIVE page_link with this emotion class rather
   than aria-current; target it so the active item gets the full theme treatment. */
[data-testid="stSidebarUserContent"] a.st-emotion-cache-1304w8u {
  background:linear-gradient(92deg, rgba(0,229,255,.18), rgba(124,92,252,.14)) !important;
  border-color:rgba(0,229,255,.32) !important;
  box-shadow: inset 3px 0 0 var(--cyan), 0 6px 18px rgba(0,229,255,.10) !important;
}
[data-testid="stSidebarUserContent"] a.st-emotion-cache-1304w8u p { color:var(--cyan) !important; font-weight:600 !important; }
[data-testid="stSidebarUserContent"] a.st-emotion-cache-1304w8u [data-testid="stIconMaterial"] {
  background:linear-gradient(135deg, var(--cyan), var(--violet)) !important; border-color:transparent !important;
  color:#06070A !important; box-shadow:0 5px 16px rgba(0,229,255,.35) !important;
}

/* tech chips */
.badges { display:flex; flex-wrap:wrap; gap:9px; }
.chip {
  font-size:.71rem; font-weight:600; color:#cfeeff; padding:.36rem .75rem; border-radius:999px;
  background:rgba(0,229,255,.08); border:1px solid rgba(0,229,255,.22);
  transition: transform .15s ease, box-shadow .15s ease;
}
.chip:hover { transform: translateY(-2px); box-shadow:0 6px 16px rgba(0,229,255,.18); }
.chip.v { color:#ded2ff; background:rgba(124,92,252,.10); border-color:rgba(124,92,252,.28); }
.chip.v:hover { box-shadow:0 6px 16px rgba(124,92,252,.22); }

/* creator credit (pinned to bottom via the flex column above) */
.credit-pin { padding-top:.6rem; }
.credit-card {
  display:flex; align-items:center; gap:.65rem; padding:.6rem .7rem; border-radius:14px;
  background:var(--card); border:1px solid var(--card-brd);
}
.avatar {
  width:36px; height:36px; border-radius:50%; display:grid; place-items:center; flex:0 0 auto;
  font-family:'Space Grotesk',sans-serif; font-weight:700; color:#06070A; font-size:1rem;
  background:linear-gradient(135deg, var(--cyan), var(--violet));
  box-shadow:0 4px 14px rgba(124,92,252,.3);
}
.credit-name { font-weight:600; font-size:.86rem; color:var(--text); line-height:1.1; }
.credit-role { color:var(--muted); font-size:.7rem; margin-top:.12rem; }
</style>
"""


def inject_css() -> None:
    """Inject the global stylesheet. Call once per page."""
    st.markdown(_CSS, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# HTML component helpers
# ---------------------------------------------------------------------------
_NUM_RE = re.compile(r"^(?P<prefix>[^\d\-]*)(?P<core>-?[\d,]*\.?\d+)(?P<suffix>.*)$")


def kpi_card(label: str, value: str, *, color: str = "cyan",
             delta: str = "", icon: str = "") -> str:
    """Return HTML for a glass KPI card.

    The value is rendered as readable text (so it works with no JS), but the
    numeric part is also embedded as data-* attributes. When animate_counters()
    is present on the page, those numbers count up from zero on load.
    """
    delta_html = f'<div class="delta" style="color:{COLORS["muted"]}">{delta}</div>' if delta else ""
    icon_html = f'<span class="spark material-symbols-rounded">{icon}</span>' if icon else ""

    m = _NUM_RE.match(value.strip())
    data, cls = "", "value"
    if m:
        core = m.group("core")
        decimals = len(core.split(".")[1]) if "." in core else 0
        data = (f' data-target="{core.replace(",", "")}"'
                f' data-prefix="{m.group("prefix")}" data-suffix="{m.group("suffix")}"'
                f' data-decimals="{decimals}" data-comma="{1 if "," in core else 0}"')
        cls = "value count-val"

    return f"""
    <div class="glass kpi {color}">
      <div class="label">{label}</div>
      <div class="{cls}"{data}>{value}</div>
      {delta_html}{icon_html}
    </div>"""


_COUNTER_JS = """
<script>
(function () {
  const doc = window.parent.document;
  function fmt(v, decimals, comma) {
    let s = v.toFixed(decimals);
    if (comma) {
      const p = s.split('.');
      p[0] = parseInt(p[0], 10).toLocaleString('en-US');
      s = p.join('.');
    }
    return s;
  }
  const els = doc.querySelectorAll('.count-val:not([data-done])');
  els.forEach(function (el) {
    el.setAttribute('data-done', '1');
    const target = parseFloat(el.getAttribute('data-target'));
    const decimals = parseInt(el.getAttribute('data-decimals'), 10) || 0;
    const comma = el.getAttribute('data-comma') === '1';
    const prefix = el.getAttribute('data-prefix') || '';
    const suffix = el.getAttribute('data-suffix') || '';
    if (isNaN(target)) return;
    const dur = 1100, start = performance.now();
    function tick(now) {
      const p = Math.min(1, (now - start) / dur);
      const e = 1 - Math.pow(1 - p, 3);            // easeOutCubic
      el.textContent = prefix + fmt(target * e, decimals, comma) + suffix;
      if (p < 1) requestAnimationFrame(tick);
      else el.textContent = prefix + fmt(target, decimals, comma) + suffix;
    }
    requestAnimationFrame(tick);
  });
})();
</script>
"""


def animate_counters() -> None:
    """Inject the count-up animation. Call once, after a KPI row is rendered."""
    components.html(_COUNTER_JS, height=0)


def hero(eyebrow: str, title_html: str, subtitle: str) -> str:
    """Return HTML for a page hero header."""
    return f"""
    <div style="margin-bottom:1.4rem">
      <span class="eyebrow">{eyebrow}</span>
      <div class="hero-title">{title_html}</div>
      <div class="hero-sub">{subtitle}</div>
    </div>"""


def tier_badge(tier: str) -> str:
    """Return HTML for a colored risk-tier badge."""
    c = TIER_COLORS.get(tier, COLORS["muted"])
    return (f'<span class="tier-badge" style="background:{c}1A;color:{c};'
            f'border:1px solid {c}55">&#9679; {tier} risk</span>')
