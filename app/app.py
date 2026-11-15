"""
Firechron · Calendario histórico armonizado y pronóstico estacional de fuego (MODIS + VIIRS, 2000–2026)
NASA Space Apps Challenge 2026 — equipo BurnInHell. No es una herramienta oficial de NASA.

Correr:
    streamlit run app/app.py
"""
import csv
import datetime as dt
import io
import json
import math
import os
import zipfile

import duckdb
import folium
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pyarrow as pa
import pyarrow.parquet as pq
import requests
import shapely
import streamlit as st
from branca.element import MacroElement, Template
from folium.plugins import Draw
from shapely.geometry import box, shape
from streamlit_folium import st_folium

import noticias

# ------------------------------------------------------------------ config
APP_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_APP = os.environ.get("FIRE_APP_DATA", os.path.join(APP_DIR, "data", "app_data.parquet"))
DATA_PRON = os.path.join(APP_DIR, "data", "pronostico_app.parquet")
DATA_1KM = os.environ.get("FIRE_1KM_DATA", os.path.join(APP_DIR, "..", "data", "salida", "armonizado.parquet"))
DATA_RAW = os.environ.get("FIRE_RAW_DATA", os.path.join(APP_DIR, "..", "data", "parquet"))
REPORTES = os.path.join(APP_DIR, "data", "reportes.csv")

GRID = 0.1
AQUA_START = dt.date(2002, 7, 4)
VIIRS_START = dt.date(2012, 1, 20)
SNPP_END = dt.date(2026, 11, 2)
REF_START = 2003
MAX_RAW_ROWS = 1_500_000
MAX_MAP_POINTS = 1500

# Paleta oficial de NASA Space Apps (guía de marca)
# Paleta de datos (rampas de los gráficos)
BLUE_L, BLUE_D, RED, RED_D = "#2E96F5", "#0042A6", "#E43700", "#8E1100"
# Paleta de interfaz: grafito espacial neutro + acentos
NAVY = "#0E1117"      # fondo más oscuro / texto sobre amarillo
BG2 = "#161B22"       # inputs y botones secundarios
CARD = "#1C212B"      # tarjetas
BORDER = "#30363D"    # bordes y grillas
YELLOW = "#E2F538"    # acento principal
CORAL = "#FF8A65"     # alertas / anomalía positiva
TXT2 = "#CBD5E1"      # texto secundario y de gráficos
WHITE = "#FFFFFF"
BLUE = BORDER

FIRE_RAMP = [[0.0, NAVY], [0.15, BLUE_D], [0.4, BLUE_L], [0.65, YELLOW], [0.85, RED], [1.0, RED_D]]
RAMPS = {"Firechron": FIRE_RAMP, "Viridis": "Viridis", "Inferno": "Inferno", "Cividis": "Cividis", "YlOrRd": "YlOrRd"}
ANOM_RAMP = [[0.0, BLUE_L], [0.5, "#2A303B"], [0.8, RED], [1.0, YELLOW]]
PROB_RAMP = [[0.0, BLUE_D], [0.3, BLUE_L], [0.5, "#5E6C8F"], [0.7, RED], [1.0, YELLOW]]

PRESETS = {
    "Mendoza (provincia)": None,  # se toma el polígono de la provincia
    "Luján de Cuyo, Mendoza": noticias.GAZ["Luján de Cuyo, Mendoza"],
    "General Alvear, Mendoza": noticias.GAZ["General Alvear, Mendoza"],
    "Sierras de Córdoba": noticias.GAZ["Sierras de Córdoba"],
    "Delta del Paraná": noticias.GAZ["Delta del Paraná"],
    "Pantanal": noticias.GAZ["Pantanal"],
    "Amazonía (arco de deforestación)": noticias.GAZ["Amazonía (arco de deforestación)"],
    "Sudamérica completa": (-56.0, 13.0, -82.0, -34.0),
}

ACK = ("We acknowledge the use of data and/or imagery from NASA's Fire Information for Resource "
       "Management System (FIRMS) (https://earthdata.nasa.gov/firms), part of NASA's Land, Atmosphere "
       "Near real-time Capability for Earth observations (LANCE) (https://earthdata.nasa.gov/lance) "
       "and NASA's Earth Science Data and Information System (ESDIS).")

import base64 as _b64
_ASSETS = os.path.join(APP_DIR, "assets")


def _img64(name):
    with open(os.path.join(_ASSETS, name), "rb") as fh:
        return _b64.b64encode(fh.read()).decode()


LOGO_IMG = _img64("logo_firechron.png")

LOGO = f"""<svg viewBox="0 0 64 64" width="{{s}}" height="{{s}}" xmlns="http://www.w3.org/2000/svg" aria-label="Firechron">
<circle cx="32" cy="32" r="27" fill="none" stroke="{WHITE}" stroke-width="3.5"/>
<g stroke="{WHITE}" stroke-width="2.5" stroke-linecap="round">
<line x1="32" y1="5" x2="32" y2="10"/><line x1="59" y1="32" x2="54" y2="32"/>
<line x1="32" y1="59" x2="32" y2="54"/><line x1="5" y1="32" x2="10" y2="32"/></g>
<path d="M32 15c3 7 11 11 11 21a11 11 0 0 1-22 0c0-5 3-8 5-10 0 4 2 6 4 6-1-6 0-11 2-17z" fill="{YELLOW}"/>
<path d="M32 33c2 3 5 5 5 8a5 5 0 0 1-10 0c0-3 3-5 5-8z" fill="{RED}"/></svg>"""

st.set_page_config(page_title="Firechron · El calendario del fuego en Sudamérica",
                   page_icon=os.path.join(APP_DIR, "assets", "favicon.png"),
                   layout="wide", initial_sidebar_state="collapsed")

# ------------------------------------------------------------------ idioma
if "lang" not in st.session_state:
    st.session_state.lang = "es"
L = st.session_state.lang


def tr(es, en):
    return es if L == "es" else en


MONTHS = (["Ene", "Feb", "Mar", "Abr", "May", "Jun", "Jul", "Ago", "Sep", "Oct", "Nov", "Dic"] if L == "es" else
          ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"])
MONTHS_L = (["enero", "febrero", "marzo", "abril", "mayo", "junio", "julio", "agosto", "septiembre", "octubre",
             "noviembre", "diciembre"] if L == "es" else
            ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October",
             "November", "December"])

# ------------------------------------------------------------------ estilo
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Fira+Sans:wght@400;500;600;700;800&family=Fira+Sans+Condensed:wght@600;700;800&display=swap');
html, body, [class*="css"], .stApp, button, input, textarea, select {{ font-family: 'Fira Sans', sans-serif; }}
.stApp {{ background: linear-gradient(135deg, {BG2} 0%, {NAVY} 100%) !important; color: {WHITE} !important; }}
[data-testid="stHeader"] {{ background: transparent; }}
[data-testid="stSidebar"] {{ background: {CARD}; border-right: 1px solid {BORDER}; }}
[data-testid="stSidebar"], [data-testid="stSidebarCollapsedControl"] {{ display: none; }}
.block-container {{ padding-top: 3.6rem; padding-bottom: 3rem; max-width: 1420px; }}
h1, h2, h3, h4 {{ font-family: 'Fira Sans Condensed', 'Fira Sans', sans-serif; letter-spacing: .2px; }}
[class*="st-key-card"] {{ background: {CARD}; border: 1px solid {BORDER}; border-radius: 10px; padding: 14px 16px 12px; }}
.fc-ct {{ font-family: 'Fira Sans Condensed', sans-serif; font-weight: 700; font-size: 1.02rem; text-transform: uppercase;
          letter-spacing: .4px; color: {WHITE}; margin: 0 0 2px; }}
.fc-ct b {{ color: {YELLOW}; margin-right: 6px; }}
.fc-hero {{ display: flex; align-items: center; gap: 18px; }}
.fc-name {{ font-family: 'Fira Sans Condensed', sans-serif; font-weight: 800; font-size: 3.1rem; line-height: 1; margin: 0; }}
.fc-name span {{ color: {YELLOW}; }}
.fc-tag {{ font-family: 'Fira Sans Condensed', sans-serif; font-weight: 700; font-size: 1.25rem; color: {YELLOW};
           text-transform: uppercase; margin-top: 6px; line-height: 1.15; }}
.fc-tag i {{ color: {WHITE}; font-style: normal; }}
.fc-team {{ margin-top: 8px; font-size: .92rem; opacity: .9; }}
.fc-team b {{ color: {YELLOW}; }}
.fc-pitch {{ font-size: 1.05rem; margin: 10px 0 4px; max-width: 900px; }}
.fc-steps {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; margin: 14px 0 18px; }}
.fc-step {{ background: {CARD}; border: 1px solid {BORDER}; border-radius: 10px; padding: 12px;
            display: flex; gap: 10px; align-items: center; }}
.fc-step .ic {{ flex: 0 0 44px; height: 44px; border-radius: 50%; background: {YELLOW}; color: {NAVY};
                display: flex; align-items: center; justify-content: center; font-size: 1.4rem; font-weight: 800;
                font-family: 'Fira Sans Condensed', sans-serif; }}
.fc-step .tx {{ font-size: .93rem; line-height: 1.2; }}
.fc-step .tx b {{ display: block; color: {YELLOW}; font-family: 'Fira Sans Condensed', sans-serif; font-size: .8rem; }}
.fc-kpis {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; margin-bottom: 12px; }}
.fc-kpi {{ background: {CARD}; border: 1px solid {BORDER}; border-radius: 10px; padding: 14px 16px; }}
.fc-kpi .l {{ font-size: .95rem; color: {TXT2}; }}
.fc-kpi .v {{ font-family: 'Fira Sans Condensed', sans-serif; font-weight: 800; font-size: 2.1rem; line-height: 1.1; }}
.fc-kpi .d {{ font-size: 1.15rem; font-weight: 700; }}
.up {{ color: {CORAL}; }} .down {{ color: #8AB4F8; }} .ok {{ color: #7CF29A; }}
.fc-chip {{ display: inline-block; padding: 4px 12px; border-radius: 999px; background: {BG2};
            border: 1px solid {BORDER}; margin: 2px 4px 2px 0; font-size: .9rem; }}
.fc-legend {{ font-size: .82rem; opacity: .9; margin-top: 4px; }}
.fc-sw {{ display:inline-block; width: 70px; height: 9px; border-radius: 5px; vertical-align: middle; margin: 0 6px; }}
.fc-fc {{ background: {CARD}; border: 1px solid {BORDER}; border-radius: 10px; padding: 14px 16px; height: 100%; }}
.fc-fc .m {{ font-family: 'Fira Sans Condensed', sans-serif; font-weight: 800; font-size: 1.3rem; }}
.fc-bar {{ height: 10px; background: rgba(255,255,255,.15); border-radius: 6px; overflow: hidden; margin: 6px 0; }}
.fc-bar > div {{ height: 100%; border-radius: 6px; }}
[class*="st-key-row_"] [data-testid="stHorizontalBlock"] {{ flex-wrap: nowrap !important; gap: .5rem; }}
[class*="st-key-row_"] [data-testid="stColumn"] {{ min-width: 0 !important; width: auto !important; flex: 1 1 auto; }}
[class*="st-key-row_"] [data-testid="stColumn"]:last-child {{ flex: 0 0 auto; }}
[class*="st-key-row_chips"] [data-testid="stColumn"], [class*="st-key-row_chips"] [data-testid="stColumn"]:last-child {{ flex: 1 1 0; }}
.st-key-modebar button {{ min-height: 66px; border-radius: 10px !important; }}
.st-key-btn_hist button[kind="primary"], .st-key-btn_hist button[data-testid="stBaseButton-primary"] {{
    background: {YELLOW} !important; color: {NAVY} !important; border: none !important; font-weight: 700; }}
.st-key-btn_hist button[kind="secondary"], .st-key-btn_hist button[data-testid="stBaseButton-secondary"] {{
    background: {CARD} !important; border: 1px solid {BORDER} !important; color: {WHITE} !important; }}
.st-key-btn_fc button[kind="primary"], .st-key-btn_fc button[data-testid="stBaseButton-primary"] {{
    background: {CORAL} !important; color: {NAVY} !important; border: none !important; font-weight: 700; }}
.st-key-btn_fc button[kind="primary"] p, .st-key-btn_hist button[kind="primary"] p {{ color: {NAVY} !important; }}
.st-key-modebar button p {{ font-size: 1.2rem !important; font-weight: 700; font-family: 'Fira Sans Condensed', sans-serif; }}
.st-key-modebar {{ margin: 4px 0 14px; }}
.st-key-btn_fc button[kind="secondary"], .st-key-btn_fc button[data-testid="stBaseButton-secondary"] {{
    background: {CARD} !important; border: 1.5px solid {CORAL} !important; color: {WHITE} !important;
    animation: fcpulse 2.4s ease-in-out infinite; }}
@keyframes fcpulse {{ 0%,100% {{ box-shadow: 0 0 0 0 rgba(255,138,101,.45); }} 50% {{ box-shadow: 0 0 0 8px rgba(255,138,101,0); }} }}
.st-key-ctrlbar {{ background: {CARD}; border: 1px solid {BORDER}; border-radius: 10px; padding: 8px 12px; }}
.fc-tip {{ background: rgba(226,245,56,.08); border: 1px dashed {YELLOW}; border-radius: 10px; padding: 8px 12px; margin: 6px 0 4px;
           font-size: .95rem; }}
.fc-foot {{ font-size: .8rem; opacity: .75; margin-top: 30px; }}
button[kind="primary"], button[data-testid="stBaseButton-primary"] {{ color: {NAVY} !important; font-weight: 700; }}
[data-baseweb="input"], [data-baseweb="select"] > div, [data-baseweb="textarea"], [data-baseweb="base-input"],
[data-testid="stNumberInput"] input, [data-testid="stDateInput"] input {{
    background: {BG2} !important; border-color: {BORDER} !important; color: {WHITE} !important; }}
[data-baseweb="input"]:hover, [data-baseweb="select"] > div:hover, [data-baseweb="input"]:focus-within {{ border-color: {YELLOW} !important; }}
button[kind="secondary"], button[data-testid="stBaseButton-secondary"], [data-testid="stPopover"] button,
[data-testid="stBaseButton-segmented_control"], [data-testid="stDownloadButton"] button {{
    background: {BG2} !important; border: 1px solid {BORDER} !important; color: {WHITE} !important; }}
button[kind="secondary"]:hover, button[data-testid="stBaseButton-secondary"]:hover, [data-testid="stPopover"] button:hover,
[data-testid="stBaseButton-segmented_control"]:hover {{ border-color: {YELLOW} !important; }}
[data-testid="stDownloadButton"] button[kind="primary"] {{ background: {YELLOW} !important; color: {NAVY} !important; border: none !important; }}
[data-testid="stExpander"] details {{ background: {CARD}; border: 1px solid {BORDER} !important; border-radius: 10px; }}
.fc-logo {{ height: 64px; width: auto; display: block; }}
[data-testid="stSegmentedControl"] button[aria-checked="true"], [data-testid="stBaseButton-segmented_controlActive"] {{
    background: {YELLOW} !important; color: {NAVY} !important; font-weight: 700; }}
[data-testid="stTabs"] button p {{ font-size: 1rem; }}
@media (max-width: 640px) {{
  .fc-name {{ font-size: 2.2rem; }} .fc-tag {{ font-size: 1rem; }} .fc-logo {{ height: 40px; }}
  .fc-steps {{ grid-template-columns: 1fr 1fr; gap: 8px; }}
  .fc-step {{ padding: 8px; }} .fc-step .ic {{ flex-basis: 34px; height: 34px; font-size: 1.05rem; }}
  .fc-kpis {{ grid-template-columns: 1fr 1fr; gap: 8px; }}
  .block-container {{ padding-top: 3.2rem; }}
  .st-key-modebar button p {{ font-size: 1.05rem !important; }}
  .fc-kpi .v {{ font-size: 1.45rem; }} .fc-kpi .l {{ font-size: .8rem; }}
}}
</style>
""", unsafe_allow_html=True)


# ------------------------------------------------------------------ datos
@st.cache_resource(show_spinner="Cargando 25 años de satélites…")
def load_data():
    t = pq.read_table(DATA_APP)
    days = t.column("date").cast(pa.int32()).to_numpy()
    d64 = days.astype("datetime64[D]")
    data = {"g_r": t.column("g_r").to_numpy().astype(np.int16), "g_c": t.column("g_c").to_numpy().astype(np.int16),
            "days": days, "focos": t.column("focos").to_numpy().astype(np.float32),
            "crudo": t.column("crudo").to_numpy().astype(np.int32)}
    return data, d64.min().astype(object), d64.max().astype(object)


@st.cache_resource
def load_pron():
    if not os.path.exists(DATA_PRON):
        return None
    return pd.read_parquet(DATA_PRON)


def source_of(d):
    if d < AQUA_START:
        return "Terra"
    if d < VIIRS_START:
        return "MODIS"
    return "SNPP" if d <= SNPP_END else "NOAA20"


def cell_selection(data, wkt):
    geom = shapely.from_wkt(wkt)
    minx, miny, maxx, maxy = geom.bounds
    r0, r1 = math.floor(miny / GRID), math.floor(maxy / GRID)
    c0, c1 = math.floor(minx / GRID), math.floor(maxx / GRID)
    g_r, g_c = data["g_r"], data["g_c"]
    idx = np.flatnonzero((g_r >= r0) & (g_r <= r1) & (g_c >= c0) & (g_c <= c1))
    if idx.size == 0:
        return idx, False
    rr, cc = g_r[idx].astype(np.int32), g_c[idx].astype(np.int32)
    key = (rr - r0) * (c1 - c0 + 1) + (cc - c0)
    ukey, inv = np.unique(key, return_inverse=True)
    ur, uc = ukey // (c1 - c0 + 1) + r0, ukey % (c1 - c0 + 1) + c0
    inside = shapely.contains_xy(geom, (uc + 0.5) * GRID, (ur + 0.5) * GRID)
    small = False
    if not inside.any():
        inside = shapely.intersects(geom, shapely.box(uc * GRID, ur * GRID, (uc + 1) * GRID, (ur + 1) * GRID))
        small = True
    return idx[inside[inv]], small


@st.cache_data(show_spinner=False, max_entries=16)
def zone_daily(wkt):
    data, dmin, dmax = load_data()
    idx, small = cell_selection(data, wkt)
    d0 = np.datetime64(dmin, "D").astype(np.int64)
    n = (np.datetime64(dmax, "D") - np.datetime64(dmin, "D")).astype(int) + 1
    if idx.size:
        dd = data["days"][idx] - d0
        f = np.bincount(dd, weights=data["focos"][idx], minlength=n)
        c = np.bincount(dd, weights=data["crudo"][idx], minlength=n)
    else:
        f = c = np.zeros(n)
    df = pd.DataFrame({"date": pd.date_range(dmin, dmax, freq="D"), "focos": f, "crudo": c})
    df["year"], df["month"], df["day"] = df.date.dt.year, df.date.dt.month, df.date.dt.day
    return df, small, int(idx.size)


@st.cache_data(show_spinner=False, max_entries=16)
def zone_cells(wkt, d0, d1):
    data, _, _ = load_data()
    idx, _ = cell_selection(data, wkt)
    days = data["days"][idx]
    a, b = np.datetime64(d0, "D").astype(np.int64), np.datetime64(d1, "D").astype(np.int64)
    idx = idx[(days >= a) & (days <= b)]
    return pd.DataFrame({"g_r": data["g_r"][idx], "g_c": data["g_c"][idx], "focos": data["focos"][idx],
                         "days": data["days"][idx]})


@st.cache_data(show_spinner=False, ttl=6 * 3600)
def buscar(q, lang):
    return [(lab, shapely.to_wkt(g, rounding_precision=4), short) for lab, g, short in noticias.buscar_lugar(q, lang)]


@st.cache_data(show_spinner=False, ttl=6 * 3600)
def _news_cached(places, countries, y, m, d1max):
    a = dt.date(y, m, 1)
    b = min(dt.date(y, m, pd.Period(f"{y}-{m:02d}").days_in_month), d1max)
    arts, err = noticias.google_news(list(places), list(countries), a, b)
    if not arts and a >= noticias.GDELT_START:
        g, err2 = noticias.gdelt_search(noticias.gdelt_query(list(places)), a, b, retries=1)
        arts = noticias.filtrar_medios(g, list(countries), maximo=8)
        err = err if (err and err2) else None
    if err and not arts:
        raise ConnectionError(err)   # no guardar en caché un fallo de red
    return arts


def news_for(places, countries, y, m, d1max):
    try:
        return _news_cached(places, countries, y, m, d1max), None
    except ConnectionError:
        return [], "net"


@st.cache_data(show_spinner=False, ttl=24 * 3600)
def wiki_for(provs, countries, y):
    return noticias.wiki_events([tuple(p) for p in provs], list(countries), y, y, max_items=6)


def ref_years(dmax):
    return REF_START, (dmax.year if dmax == dt.date(dmax.year, 12, 31) else dmax.year - 1)


def monthly_table(daily, d0, d1, ref):
    ra, rb = ref
    full = daily.groupby(["year", "month"], as_index=False)[["focos", "crudo"]].sum()
    clim = full[(full.year >= ra) & (full.year <= rb)].groupby("month").focos.median()
    sel = daily[(daily.date.dt.date >= d0) & (daily.date.dt.date <= d1)]
    mt = sel.groupby(["year", "month"], as_index=False).agg(focos=("focos", "sum"), crudo=("crudo", "sum"),
                                                          ndays=("focos", "size"))
    mt["mlen"] = [pd.Period(f"{y}-{m:02d}").days_in_month for y, m in zip(mt.year, mt.month)]
    mt["partial"] = mt.ndays < mt.mlen
    mt["median"] = mt.month.map(clim)
    mt["anom"] = np.where(mt["median"] > 0, (mt.focos - mt["median"]) / mt["median"] * 100, np.nan)
    mt["source"] = [source_of(dt.date(y, m, 15)) for y, m in zip(mt.year, mt.month)]
    return mt, clim, full


def compare_month(daily, y, m, dmax, ref):
    ra, rb = ref
    mlen = pd.Period(f"{y}-{m:02d}").days_in_month
    D = dmax.day if (y, m) == (dmax.year, dmax.month) else mlen
    cur = daily[(daily.year == y) & (daily.month == m) & (daily.day <= D)].focos.sum()
    vals = {yy: daily[(daily.year == yy) & (daily.month == m) & (daily.day <= D)].focos.sum()
            for yy in range(ra, rb + 1) if yy != y}
    v = np.array(list(vals.values()))
    med = float(np.median(v)) if v.size else np.nan
    return dict(D=D, cur=float(cur), med=med, p25=float(np.percentile(v, 25)), p75=float(np.percentile(v, 75)),
                rank=int(np.sum(v > cur)) + 1, n=v.size + 1, vals=vals,
                anom=(cur - med) / med * 100 if med > 0 else np.nan)


def fmt(x, dec=0):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "–"
    s = f"{x:,.{dec}f}"
    return s.replace(",", "X").replace(".", ",").replace("X", ".") if L == "es" else s


def anom_words(a):
    """Anomalía en palabras simples."""
    if a is None or np.isnan(a):
        return tr("sin referencia", "no reference"), "ok"
    r = 1 + a / 100
    if a >= 15:
        return (tr(f"{fmt(r, 1)} veces lo normal", f"{fmt(r, 1)}× normal") if r >= 2 else
                tr(f"{a:+.0f} % más de lo normal", f"{a:+.0f}% above normal")), "up"
    if a <= -15:
        return tr(f"{abs(a):.0f} % menos de lo normal", f"{abs(a):.0f}% below normal"), "down"
    return tr("dentro de lo normal", "about normal"), "ok"


def style_fig(fig, h, **kw):
    fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      font=dict(family="Fira Sans, sans-serif", color=TXT2, size=13), height=h,
                      **{"margin": dict(l=10, r=10, t=50, b=10), **kw})
    fig.update_xaxes(gridcolor=BORDER, fixedrange=True)
    fig.update_yaxes(gridcolor=BORDER, fixedrange=True)
    fig.update_layout(dragmode=False)
    return fig


def ramp_color(scale, v):
    """Color de una rampa plotly (lista o nombre) en v∈[0,1]."""
    import plotly.colors as pc
    if isinstance(scale, str):
        scale = pc.get_colorscale(scale)
    return pc.sample_colorscale(scale, [min(1, max(0, float(v)))])[0]


def card_title(num, text, help_md=None):
    c1, c2 = st.container(key=f"row_t{abs(hash(num)) % 10**6}").columns([7, 1], vertical_alignment="center")
    c1.markdown(f'<div class="fc-ct"><b>{num}</b>{text}</div>', unsafe_allow_html=True)
    if help_md:
        with c2.popover(tr("Info", "Info"), help=tr("¿Qué es esto?", "What is this?")):
            st.markdown(help_md)


def help_pop(md, label="?"):
    with st.popover(label):
        st.markdown(md)


def readme(desc, units):
    return (f"{desc}\n\nUnits / Unidades: {units}\n\n"
            "Method / Método: MODIS (Terra+Aqua, 1 km) and VIIRS (S-NPP, NOAA-20, 375 m) active-fire hotspots from "
            "NASA FIRMS, filtered (nominal/high confidence, vegetation fires, static sources removed), counted as "
            "1-km cell-days and harmonized to the VIIRS S-NPP scale with region-specific (1°x1°) factors calibrated "
            "on 2012-2017 and validated on 2018-2020.\n\n"
            f"Data acknowledgment:\n{ACK}\n\nFirechron — team BurnInHell, NASA Space Apps Challenge 2026 (Mendoza). "
            "Not an official NASA product.\n")


def zip_bytes(name, df, desc, units):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr(f"{name}.csv", df.to_csv(index=False))
        z.writestr("LEEME_README.txt", readme(desc, units))
    return buf.getvalue()


def set_zone(wkt, label, short):
    st.session_state.zone = {"wkt": wkt, "label": label, "short": short}
    st.session_state.zone_toast = label
    st.session_state.pop("news_open", None)


# ------------------------------------------------------------------ datos base
try:
    data, DMIN, DMAX = load_data()
except FileNotFoundError:
    st.error(f"No encuentro los datos de la app: {DATA_APP}. Corré primero `python app/preparar_datos_app.py`.")
    st.stop()
REF = ref_years(DMAX)
PRON = load_pron()

if "zone" not in st.session_state:
    mdz = [r for r in buscar("Mendoza", "es") if r[0].startswith("Mendoza")]
    if mdz:
        set_zone(mdz[0][1], "Mendoza, Argentina", "Mendoza")
    else:
        a, b, c, d = noticias.GAZ["Luján de Cuyo, Mendoza"]
        set_zone(shapely.to_wkt(box(c, a, d, b)), "Luján de Cuyo, Mendoza", "Luján de Cuyo")
st.session_state.setdefault("map_v", 0)
st.session_state.setdefault("show_pts", True)  # focos visibles por defecto

# ================================================================== 1. CABECERA
h1, h2 = st.columns([8, 2], vertical_alignment="top")
with h1:
    st.markdown(f"""
<div class="fc-hero">
<div><img class="fc-logo" src="data:image/png;base64,{LOGO_IMG}" alt="Firechron">
<div class="fc-tag">{tr('Calendario histórico armonizado <i>y</i> pronóstico estacional de fuego',
                         'Harmonized fire history calendar <i>and</i> seasonal fire outlook')}
<i>· MODIS + VIIRS, 2000–2026</i></div>
<div class="fc-team">{tr('Equipo', 'Team')} <b>Firechron</b> · NASA Space Apps Challenge 2026 · Mendoza, Argentina</div></div></div>
<div class="fc-pitch">{tr('Convertimos <b>25 años de satélites de la NASA</b> en un calendario simple: <b>cuándo y dónde se quema Sudamérica</b>, cómo viene este mes y qué esperar en los <b>próximos 3 meses</b>.',
                          'We turn <b>25 years of NASA satellite data</b> into a simple calendar: <b>when and where South America burns</b>, how this month compares and what to expect over the <b>next 3 months</b>.')}</div>
""", unsafe_allow_html=True)
with h2:
    with st.popover(tr("Idioma · Info", "Language · Info"), width="stretch"):
        lang = st.segmented_control(tr("Idioma", "Language"), ["Español", "English"],
                                    default="Español" if L == "es" else "English")
        if lang and (("es" if lang == "Español" else "en") != L):
            st.session_state.lang = "es" if lang == "Español" else "en"
            st.rerun()
        st.markdown(tr("**Firechron** es un proyecto del equipo BurnInHell para el NASA Space Apps Challenge 2026. "
                       "No es una herramienta oficial de NASA.",
                       "**Firechron** is a project by team BurnInHell for the NASA Space Apps Challenge 2026. "
                       "Not an official NASA tool."))
        st.markdown("[GitHub](https://github.com/lautarodiazopazo/firecal)")

# ================================================================== 2. INSTRUCTIVO
st.markdown(f"""
<div class="fc-steps">
 <div class="fc-step"><div class="ic">1</div><div class="tx">{tr('Buscá un lugar o dibujá una zona', 'Search a place or draw an area')}</div></div>
 <div class="fc-step"><div class="ic">2</div><div class="tx">{tr('Elegí: ver la historia o el pronóstico', 'Choose: history or outlook')}</div></div>
 <div class="fc-step"><div class="ic">3</div><div class="tx">{tr('Tocá un mes del calendario: datos y noticias', 'Tap a month: numbers and news')}</div></div>
 <div class="fc-step"><div class="ic">4</div><div class="tx">{tr('Descargá los datos en CSV', 'Download the data as CSV')}</div></div>
</div>""", unsafe_allow_html=True)

# ================================================================== 3. PANEL PRINCIPAL
st.session_state.setdefault("fc_mode", False)
with st.container(key="modebar"):
    mb1, mb2 = st.columns(2)
    _ya, _yb = st.session_state.get("yrs", (DMIN.year, DMAX.year))
    _per = f"{_ya}–{_yb}" if _ya != _yb else f"{_ya}"
    if mb1.button(tr(f"Ver la historia · {_per}", f"See the history · {_per}"), key="btn_hist",
                  type="secondary" if st.session_state.fc_mode else "primary", width="stretch"):
        st.session_state.fc_mode = False
        st.rerun()
    if mb2.button(tr("Predecir los próximos 3 meses", "Forecast the next 3 months"), key="btn_fc",
                  type="primary" if st.session_state.fc_mode else "secondary", width="stretch"):
        st.session_state.fc_mode = True
        st.rerun()
colL, colM = st.columns([1.1, 3.2], gap="medium")
with colM:
    kpi_box = st.container()
    ctrl_box = st.container(key="ctrlbar")
    map_box = st.container()
    cal_box = st.container()

# ---------- 01 Área
with colL:
    with st.container(key="card_area"):
        card_title("01", tr("Área de interés", "Area of interest"),
                   tr("Escribí el nombre de un **lugar** (departamento, ciudad, provincia, parque…) o **dibujá** "
                      "un rectángulo o polígono en el mapa con los botones ▭ ⬠ de la izquierda. "
                      "Se analiza **una zona por vez**: si dibujás otra, reemplaza a la anterior.",
                      "Type a **place** name (district, city, province, park…) or **draw** a rectangle or polygon "
                      "on the map with the ▭ ⬠ buttons. **One area at a time**: a new drawing replaces the previous one."))
        q = st.text_input(tr("Buscar lugar", "Search place"), placeholder=tr("🔍  Ej.: Luján de Cuyo", "🔍  e.g. Pantanal"),
                          label_visibility="collapsed", key="q")
        if q:
            res = buscar(q, L)
            if not res:
                st.caption(tr("No encontré ese lugar. Probá con otro nombre o dibujalo en el mapa.",
                              "Place not found. Try another name or draw it on the map."))
            else:
                labels = [r[0] for r in res]
                pick = st.selectbox(tr("Resultados", "Results"), range(len(res)), format_func=lambda i: labels[i],
                                    label_visibility="collapsed", key=f"pick_{q}")
                if st.session_state.get("applied_q") != (q, pick):
                    st.session_state.applied_q = (q, pick)
                    set_zone(res[pick][1], res[pick][0], res[pick][2])
        how = st.segmented_control(tr("Otras formas", "Other ways"),
                                   [tr("Dibujar en mapa", "Draw on map"), tr("Coordenadas", "Coordinates"),
                                    tr("Ejemplos", "Examples")], label_visibility="collapsed", key="how")
        if how == tr("Dibujar en mapa", "Draw on map"):
            st.caption(tr("Usá ▭ (rectángulo) o ⬠ (polígono) arriba a la izquierda del mapa.",
                          "Use ▭ (rectangle) or ⬠ (polygon) at the top-left of the map."))
        elif how == tr("Coordenadas", "Coordinates"):
            c1, c2 = st.columns(2)
            la0 = c1.number_input(tr("Lat. mín.", "Min lat"), -60.0, 15.0, -33.6, 0.1)
            la1 = c2.number_input(tr("Lat. máx.", "Max lat"), -60.0, 15.0, -32.9, 0.1)
            lo0 = c1.number_input(tr("Lon. mín.", "Min lon"), -85.0, -30.0, -69.75, 0.1)
            lo1 = c2.number_input(tr("Lon. máx.", "Max lon"), -85.0, -30.0, -68.75, 0.1)
            if st.button(tr("Usar estas coordenadas", "Use these coordinates"), width="stretch"):
                if la1 > la0 and lo1 > lo0:
                    set_zone(shapely.to_wkt(box(lo0, la0, lo1, la1)), f"{la0}, {lo0} → {la1}, {lo1}", "")
                else:
                    st.warning(tr("El mínimo tiene que ser menor que el máximo.", "Min must be lower than max."))
        elif how == tr("Ejemplos", "Examples"):
            ex = st.selectbox(tr("Zona de ejemplo", "Example area"), ["—"] + list(PRESETS), key="ex")
            if ex != "—" and st.session_state.get("applied_ex") != ex:
                st.session_state.applied_ex = ex
                if PRESETS[ex] is None:
                    r = [x for x in buscar("Mendoza", "es") if x[0].startswith("Mendoza")]
                    set_zone(r[0][1], "Mendoza, Argentina", "Mendoza")
                else:
                    a, b, c, d = PRESETS[ex]
                    set_zone(shapely.to_wkt(box(c, a, d, b)), ex, ex.split(" (")[0].split(",")[0])
        st.markdown(f"<div class='fc-legend'>{tr('Zona', 'Area')}: <b>{st.session_state.zone['label']}</b></div>",
                    unsafe_allow_html=True)

# ---------- 02 Período
FORECAST = bool(st.session_state.fc_mode) and PRON is not None
with colL:
    with st.container(key="card_mode"):
        card_title("02", tr("Período", "Period"),
                   tr("Elegí los **años** que querés ver. Todo (mapa, números y calendario) se actualiza solo.",
                      "Choose the **years** to see. Everything (map, numbers, calendar) updates automatically."))
        years_all = list(range(DMIN.year, DMAX.year + 1))
        y0, y1 = st.select_slider(tr("Años", "Years"), options=years_all, value=(years_all[0], years_all[-1]), key="yrs",
                                  label_visibility="collapsed")
        D0, D1 = max(dt.date(y0, 1, 1), DMIN), min(dt.date(y1, 12, 31), DMAX)
        if FORECAST:
            fmonths = PRON[["year", "month", "lead"]].drop_duplicates().sort_values("lead").values.tolist()
            fsel = st.segmented_control(tr("Mes a pronosticar", "Month to forecast"), list(range(len(fmonths))),
                                        format_func=lambda i: f"{MONTHS[fmonths[i][1] - 1]} {fmonths[i][0]}",
                                        default=0, key="fsel") or 0
            FY, FM, FL = fmonths[fsel]
    news_box = st.container(key="card_news")

# ---------- barra de controles del mapa (cerca de lo que cambian)
with ctrl_box:
    cb1, cb2, cb3, cb4 = st.columns([2.6, 1.3, 1.0, 1.1], vertical_alignment="center")
    cmode = cb1.segmented_control(tr("Ver", "Show"), [tr("Cantidad de focos", "Hotspot count"),
                                                     tr("Comparado con lo normal", "Vs. normal")],
                                  default=tr("Cantidad de focos", "Hotspot count"), key="cmode",
                                  label_visibility="collapsed") or tr("Cantidad de focos", "Hotspot count")
    ANOM = cmode != tr("Cantidad de focos", "Hotspot count")
    if not FORECAST:
        cb2.toggle(tr("Ver focos", "Show hotspots"), key="show_pts")
    with cb3.popover(tr("Colores", "Colors")):
        ramp = st.selectbox(tr("Rampa de color", "Color ramp"), list(RAMPS), index=0, key="ramp",
                            help=tr("En «comparado con lo normal» siempre: azul = menos fuego, rojo = más fuego.",
                                    "In «vs. normal» mode always: blue = less fire, red = more fire."))
    if cb4.button(tr("Borrar zona", "Clear area"), width="stretch"):
        st.session_state.map_v += 1
        st.session_state.pop("last_draw", None)
        set_zone(shapely.to_wkt(box(-82, -56, -34, 13)), tr("Sudamérica completa", "All of South America"), "")
        st.rerun()
ramp = st.session_state.get("ramp", "Firechron")

# ---------- cálculos de la zona
GEOM = shapely.from_wkt(st.session_state.zone["wkt"])
WKT = st.session_state.zone["wkt"]
daily, small, nrows = zone_daily(WKT)
HAS = nrows > 0 and daily.focos.sum() > 0
if HAS:
    mt, clim, full = monthly_table(daily, D0, D1, REF)
    cells = zone_cells(WKT, D0, D1)
    agg = cells.groupby(["g_r", "g_c"], as_index=False).focos.sum()
    agg = agg[agg.focos > 0]
    provs, ctrs = noticias.places_for((agg.g_r + 0.5) * GRID, (agg.g_c + 0.5) * GRID, agg.focos)
else:
    mt = pd.DataFrame(columns=["year", "month", "focos", "crudo", "median", "anom", "source", "partial"])
    clim, full, agg, provs, ctrs = pd.Series(dtype=float), pd.DataFrame(), pd.DataFrame(), [], []
short = st.session_state.zone.get("short") or ""
PLACES = tuple(([short] if short else []) + [p for p, _ in provs if p != short])
COUNTRIES = tuple(ctrs or ["Argentina"])

# pronóstico de la zona
if FORECAST:
    pz = PRON[(PRON.year == FY) & (PRON.month == FM)]
    minx, miny, maxx, maxy = GEOM.bounds
    pz = pz[(pz.lon >= minx - 0.25) & (pz.lon <= maxx + 0.25) & (pz.lat >= miny - 0.25) & (pz.lat <= maxy + 0.25)]
    sq = shapely.box(pz.lon - 0.25, pz.lat - 0.25, pz.lon + 0.25, pz.lat + 0.25)
    pz = pz[shapely.intersects(GEOM, sq)] if len(pz) else pz

    def zone_prob(y, m):
        p = PRON[(PRON.year == y) & (PRON.month == m)]
        p = p.merge(pz[["lat", "lon"]], on=["lat", "lon"])
        act = p[p.p_clim > 0]
        p = act if len(act) else p
        return (float(p.prob.mean()), float(p.p_clim.mean())) if len(p) else (np.nan, np.nan)

# ---------- MAPA
with map_box:
    fmap = folium.Map(location=[-33, -66], zoom_start=4, tiles="OpenStreetMap", control_scale=True,
                      scrollWheelZoom=False, prefer_canvas=True)
    fmap.get_root().header.add_child(folium.Element(
        "<style>.leaflet-container{background:#0E1117}"
        ".leaflet-tile-pane{filter:invert(1) grayscale(1) brightness(.62) contrast(1.2)}"
        ".fc-zone{filter:drop-shadow(0 0 6px rgba(226,245,56,.7))}</style>"))
    fg = folium.FeatureGroup(name="zona").add_to(fmap)
    folium.GeoJson(shapely.geometry.mapping(GEOM), style_function=lambda _: {
        "color": YELLOW, "weight": 2.5, "fillColor": YELLOW, "fillOpacity": 0.12, "className": "fc-zone"}).add_to(fg)
    if FORECAST and len(pz):
        feats = [{"type": "Feature", "geometry": shapely.geometry.mapping(shapely.box(r.lon - .25, r.lat - .25, r.lon + .25, r.lat + .25)),
                  "properties": {"c": ramp_color(PROB_RAMP, r.prob),
                                 "t": tr(f"Prob. de más fuego de lo normal: {r.prob * 100:.0f} %",
                                         f"Chance of above-normal fire: {r.prob * 100:.0f}%")}}
                 for r in pz.itertuples()]
        folium.GeoJson({"type": "FeatureCollection", "features": feats},
                       style_function=lambda f: {"color": f["properties"]["c"], "weight": 0.5, "fillColor": f["properties"]["c"],
                                                 "fillOpacity": 0.55},
                       tooltip=folium.GeoJsonTooltip(["t"], labels=False)).add_to(fmap)
    elif HAS and len(agg) and st.session_state.get("show_pts", True):
        pts = agg.copy()
        if len(pts) > MAX_MAP_POINTS:  # agregamos a celdas más grandes para no saturar el mapa
            k = int(math.ceil(math.sqrt(len(pts) / MAX_MAP_POINTS)))
            pts["g_r"], pts["g_c"] = pts.g_r // k * k, pts.g_c // k * k
            pts = pts.groupby(["g_r", "g_c"], as_index=False).focos.sum()
            cell = GRID * k
        else:
            cell = GRID
        pts["lat"], pts["lon"] = (pts.g_r + 0.5 * cell / GRID) * GRID, (pts.g_c + 0.5 * cell / GRID) * GRID
        lv = np.sqrt(pts.focos.values)
        vmax = max(np.percentile(lv, 99), 1e-6)
        pts = pts.assign(v=np.clip(lv / vmax, 0, 1)).sort_values("v")
        feats = [{"type": "Feature", "geometry": {"type": "Point", "coordinates": [r.lon, r.lat]},
                  "properties": {"c": ramp_color(RAMPS[ramp] if not ANOM else FIRE_RAMP, 0.25 + 0.75 * r.v),
                                 "r": 1.5 + 3.5 * r.v, "t": f"{fmt(r.focos)} {tr('focos', 'hotspots')}"}}
                 for r in pts.itertuples()]
        folium.GeoJson({"type": "FeatureCollection", "features": feats},
                       marker=folium.CircleMarker(radius=4, fill=True),
                       style_function=lambda f: {"radius": f["properties"]["r"], "color": f["properties"]["c"],
                                                 "weight": 0, "fill": True, "fillColor": f["properties"]["c"], "fillOpacity": 0.85},
                       tooltip=folium.GeoJsonTooltip(["t"], labels=False)).add_to(fmap)
    minx, miny, maxx, maxy = GEOM.bounds
    fmap.fit_bounds([[miny, minx], [maxy, maxx]], padding=(20, 20))
    Draw(export=False, feature_group=fg, position="topleft",
         draw_options={"polyline": False, "circle": False, "marker": False, "circlemarker": False,
                       "polygon": {"shapeOptions": {"color": YELLOW}}, "rectangle": {"shapeOptions": {"color": YELLOW}}},
         edit_options={"edit": False, "remove": False}).add_to(fmap)
    one = MacroElement()
    one._template = Template("""{% macro script(this, kwargs) %}
        {{ this._parent.get_name() }}.on('draw:drawstart', function(){ %s.clearLayers(); });
    {% endmacro %}""".replace("%s", fg.get_name()))
    fmap.add_child(one)
    out = st_folium(fmap, height=470, use_container_width=True, key=f"map{st.session_state.map_v}",
                    returned_objects=["last_active_drawing"])
    if out and out.get("last_active_drawing"):
        g = out["last_active_drawing"].get("geometry")
        sig = json.dumps(g, sort_keys=True)
        if g and g.get("type") in ("Polygon", "MultiPolygon") and st.session_state.get("last_draw") != sig:
            st.session_state.last_draw = sig
            set_zone(shapely.to_wkt(shape(g), rounding_precision=4), tr("Zona dibujada", "Drawn area"), "")
            st.rerun()
    lc1 = st.container()
    lc1.markdown(f"<div class='fc-legend' style='color:{YELLOW}'>{tr('Debajo: el calendario de 25 años, los gráficos y las descargas', 'Below: the 25-year calendar, charts and downloads')}</div>",
                 unsafe_allow_html=True)
    if FORECAST:
        grad = "linear-gradient(90deg," + ",".join(c for _, c in PROB_RAMP) + ")"
        lc1.markdown(f"<div class='fc-legend'>{tr('Probabilidad de más fuego de lo normal', 'Chance of above-normal fire')}: "
                     f"0 %<span class='fc-sw' style='background:{grad}'></span>100 % · {tr('cuadros de ~50 km', '~50 km squares')}</div>",
                     unsafe_allow_html=True)
    elif st.session_state.get("show_pts", True):
        grad = "linear-gradient(90deg," + ",".join(c for _, c in FIRE_RAMP[1:]) + ")"
        lc1.markdown(f"<div class='fc-legend'>{tr('Cada punto = un área de ~11 km', 'Each dot = a ~11 km area')} · "
                     f"{tr('pocos', 'few')}<span class='fc-sw' style='background:{grad}'></span>{tr('muchos focos', 'many hotspots')} "
                     f"({D0.year}–{D1.year})</div>", unsafe_allow_html=True)

# ---------- KPIs
with kpi_box:
    if not HAS:
        st.info(tr("No hay focos registrados en esta zona.", "No hotspots recorded in this area."))
    elif FORECAST:
        p, pc_ = zone_prob(FY, FM)
        hist_m = full[(full.month == FM) & (full.year >= REF[0]) & (full.year <= REF[1])].focos
        diff = p - pc_
        verdict, cls = ((tr("Más fuego de lo normal", "Above normal"), "up") if diff > 0.08 else
                        (tr("Menos fuego de lo normal", "Below normal"), "down") if diff < -0.08 else
                        (tr("Dentro de lo normal", "About normal"), "ok"))
        st.markdown(f"""<div class="fc-kpis">
<div class="fc-kpi"><div class="l">{tr('Pronóstico para', 'Outlook for')} {MONTHS_L[FM - 1]} {FY}</div><div class="v {cls}" style="font-size:1.55rem">{verdict}</div></div>
<div class="fc-kpi"><div class="l">{tr('Probabilidad de un mes con más fuego de lo normal', 'Chance of a month with above-normal fire')}</div><div class="v">{p * 100:.0f} %</div></div>
<div class="fc-kpi"><div class="l">{tr('Lo habitual (2003–2019)', 'Usual (2003–2019)')}</div><div class="v">{pc_ * 100:.0f} %</div><div class="l">{tr('de los años', 'of years')}</div></div>
<div class="fc-kpi"><div class="l">{tr('Focos en un', 'Hotspots in a typical')} {MONTHS_L[FM - 1]} {tr('típico', '')}</div><div class="v">{fmt(hist_m.median())}</div></div>
</div>""", unsafe_allow_html=True)
        help_pop(tr("**¿Qué predice el modelo?** Si el mes elegido va a tener **más focos que la mediana** de ese mismo "
                    "mes en 2003–2019, en cuadros de ~50 km.\n\n**¿Con qué?** Con la historia de fuego reciente de la zona y "
                    "de sus vecinos, el clima (ERA5-Land) y El Niño/La Niña (índice ONI).\n\n**¿Qué tan confiable es?** "
                    "Evaluado en 2023–2025, años que no vio al entrenar: acierta en **~7 de cada 10** casos a 1 mes y "
                    "**~2 de cada 3** a 3 meses. Usar sólo «lo habitual» acierta ~6 de cada 10.\n\n"
                    "Es una **orientación** para prepararse, no una certeza.",
                    "**What does it predict?** Whether the chosen month will have **more hotspots than the median** of "
                    "that month in 2003–2019, in ~50 km squares.\n\n**Using?** Recent fire history of the area and its "
                    "neighbors, climate (ERA5-Land) and El Niño/La Niña (ONI index).\n\n**How reliable?** Tested on "
                    "2023–2025 (unseen in training): right **~7 in 10** times at 1 month and **~2 in 3** at 3 months. "
                    "Using only «the usual» is right ~6 in 10.\n\nIt is **guidance** to prepare, not a certainty."),
                 tr("¿Cómo leer esto?", "How to read this?"))
    else:
        yr = mt.groupby("year").focos.sum()
        cmp_now = compare_month(daily, DMAX.year, DMAX.month, DMAX, REF)
        aw, acls = anom_words(cmp_now["anom"])
        peak = MONTHS_L[int(clim.idxmax()) - 1].capitalize() if clim.max() > 0 else "–"
        st.markdown(f"""<div class="fc-kpis">
<div class="fc-kpi"><div class="l">{tr('Focos en el período', 'Hotspots in period')} ({D0.year}–{D1.year})</div><div class="v">{fmt(mt.focos.sum())}</div></div>
<div class="fc-kpi"><div class="l">{tr('Mes típicamente más activo', 'Typically busiest month')}</div><div class="v">{peak}</div></div>
<div class="fc-kpi"><div class="l">{tr('Año más activo', 'Busiest year')}</div><div class="v">{int(yr.idxmax()) if len(yr) else '–'}</div><div class="l">{fmt(yr.max()) if len(yr) else ''} {tr('focos', 'hotspots')}</div></div>
<div class="fc-kpi"><div class="l">{MONTHS_L[DMAX.month - 1].capitalize()} {DMAX.year} {tr('vs. lo normal', 'vs. normal')} <small>(1–{cmp_now['D']})</small></div><div class="v">{fmt(cmp_now['cur'])} <span class="d {acls}">{'' if np.isnan(cmp_now['anom']) else f"({cmp_now['anom']:+.0f} %)"}</span></div><div class="l {acls}">{aw}</div></div>
</div>""", unsafe_allow_html=True)
        help_pop(tr("**Foco de calor**: un lugar de ~1 km donde un satélite de la NASA detectó fuego activo en un día. "
                    "Ya está **armonizado**: cuenta igual aunque lo haya visto un satélite viejo (MODIS) o uno nuevo (VIIRS).\n\n"
                    "**Lo normal** = la **mediana** de 2003–2025: el valor del medio, la mitad de los años tuvo menos y la otra mitad más.\n\n"
                    "**+34 %** quiere decir 34 % más focos que en un año normal; **−20 %**, 20 % menos.",
                    "**Hotspot**: a ~1 km spot where a NASA satellite detected active fire on a given day. It is "
                    "**harmonized**: it counts the same whether an old (MODIS) or new (VIIRS) satellite saw it.\n\n"
                    "**Normal** = the **median** of 2003–2025: half the years had less, half had more.\n\n"
                    "**+34%** means 34% more hotspots than a normal year; **−20%**, 20% fewer."),
                 tr("¿Qué significan estos números?", "What do these numbers mean?"))
    if small:
        st.caption(tr("Zona muy chica: se usan los cuadros de ~11 km que la tocan.", "Very small area: ~11 km cells touching it are used."))


# ================================================================== detalle de un mes (ventana)
def month_detail(y, m):
    @st.dialog(f"{MONTHS_L[m - 1].capitalize()} {y} · {st.session_state.zone['label']}", width="large")
    def _dlg():
        C = compare_month(daily, y, m, DMAX, REF)
        aw, acls = anom_words(C["anom"])
        src = source_of(dt.date(y, m, 15))
        st.markdown(f"""<div class="fc-kpis" style="grid-template-columns:repeat(3,1fr)">
<div class="fc-kpi"><div class="l">{tr('Focos', 'Hotspots')}{'' if C['D'] >= 28 else f' (1–{C["D"]})'}</div><div class="v">{fmt(C['cur'])}</div></div>
<div class="fc-kpi"><div class="l">{tr('En un', 'In a normal')} {MONTHS_L[m - 1]} {tr('normal', '')}</div><div class="v">{fmt(C['med'])}</div></div>
<div class="fc-kpi"><div class="l">{tr('Comparado', 'Compared')}</div><div class="v {acls}" style="font-size:1.35rem">{aw}</div></div></div>""",
                    unsafe_allow_html=True)
        st.markdown(tr(f"Fue el **{C['rank']}.º {MONTHS_L[m - 1]} con más fuego** de {C['n']} años registrados.",
                       f"It was the **#{C['rank']} {MONTHS_L[m - 1]} with most fire** out of {C['n']} years."))
        if src == "Terra":
            st.caption(tr("⚠️ En esa época había un solo satélite (Terra): el dato es menos confiable.",
                          "⚠️ Only one satellite (Terra) back then: less reliable."))
        vy = pd.Series(C["vals"])
        vy.loc[y] = C["cur"]
        vy = vy.sort_index()
        f = go.Figure(go.Bar(x=vy.index, y=vy.values, marker_color=[RED if k == y else BLUE_L for k in vy.index],
                             hovertemplate="%{x}: %{y:,.0f}<extra></extra>"))
        f.add_hline(y=C["med"], line_dash="dash", line_color=WHITE, annotation_text=tr("lo normal", "normal"))
        st.plotly_chart(style_fig(f, 240, title=tr(f"{MONTHS_L[m - 1].capitalize()} de cada año", f"{MONTHS_L[m - 1]} of each year"),
                                  showlegend=False), width="stretch", config={"displayModeBar": False})
        st.markdown(f"#### 📰 {tr('Qué decía la prensa', 'What the press said')}")
        if not PLACES:
            st.caption(tr("No hay lugares para buscar.", "No places to search."))
        else:
            with st.spinner(tr("Buscando noticias…", "Searching news…")):
                arts, err = news_for(PLACES, COUNTRIES, y, m, DMAX)
            if not arts:
                st.caption(tr("No pudimos conectarnos al buscador de noticias. Probá de nuevo en un rato.",
                              "Could not reach the news search. Try again later.") if err else
                           tr("No encontramos notas de ese mes.", "No articles found for that month."))
            for a in arts:
                d = a.get("date", "")
                dd = f"{d[6:8]}/{d[4:6]}/{d[:4]}" if len(d) == 8 else ""
                st.markdown(f"- [{a['title']}]({a['url']})  \n  <small>{'📍 ' if a.get('origen') == 'local' else ''}{a['domain']} · {dd}</small>",
                            unsafe_allow_html=True)
            wk = wiki_for(tuple(tuple(p) for p in provs), COUNTRIES, y)
            if wk:
                st.markdown(f"**📚 {tr('Eventos destacados', 'Notable events')} ({y})**")
                for a in wk:
                    st.markdown(f"- [{a['title']}]({a['url']}) <small>Wikipedia</small>", unsafe_allow_html=True)
            st.caption(tr(f"Buscamos: {', '.join(PLACES)}. Sólo titulares y enlaces; el contenido es de cada medio. "
                          "Fuentes: Google News, Wikipedia.",
                          f"Searched: {', '.join(PLACES)}. Headlines and links only. Sources: Google News, Wikipedia."))
    _dlg()


# ================================================================== FILA 2: noticias + calendario
with news_box:
    if True:
        card_title("03", tr("Noticias y contexto del lugar", "News and context"),
                   tr("Mostramos los **lugares** donde hubo más focos en tu zona y los **meses más fuera de lo normal**. "
                      "Tocá un mes (acá o en el calendario) para ver los números y las **noticias** de ese momento.",
                      "We show the **places** with most hotspots in your area and the **most unusual months**. "
                      "Tap a month (here or in the calendar) to see numbers and **news** from that time."))
        if provs or ctrs:
            st.markdown(", ".join(f"{p}" for p, _ in provs) or ", ".join(noticias.COUNTRY_ES.get(c, c) for c in ctrs))
        if HAS and len(mt):
            cand = mt.assign(exceso=mt.focos - mt["median"].fillna(0)).sort_values("exceso", ascending=False).head(3)
            st.caption(tr("Meses con más fuego de lo normal:", "Months with most excess fire:"))
            cc = st.container(key="row_chips").columns(3)
            for i, r in enumerate(cand.itertuples()):
                if cc[i].button(f"{MONTHS[int(r.month) - 1]} {int(r.year)}", key=f"chip{i}", width="stretch"):
                    month_detail(int(r.year), int(r.month))

with cal_box:
    with st.container(key="card_cal"):
        if FORECAST:
            card_title("📈", tr("Pronóstico de los próximos 3 meses", "Next 3 months outlook"))
            fc = st.columns(len(fmonths))
            for i, (yy, mm, ll) in enumerate(fmonths):
                p, pc_ = zone_prob(yy, mm)
                col = ramp_color(PROB_RAMP, p) if not np.isnan(p) else "#666"
                fc[i].markdown(f"""<div class="fc-fc"><div class="m">{MONTHS_L[mm - 1].capitalize()} {yy}</div>
<div class="l">{tr('Prob. de más fuego de lo normal', 'Chance of above-normal fire')}</div>
<div class="fc-kpi" style="border:0;padding:0;background:none"><div class="v">{'–' if np.isnan(p) else f'{p * 100:.0f} %'}</div></div>
<div class="fc-bar"><div style="width:{0 if np.isnan(p) else p * 100:.0f}%;background:{col}"></div></div>
<div class="l">{tr('Habitual', 'Usual')}: {'–' if np.isnan(pc_) else f'{pc_ * 100:.0f} %'} · {tr('anticipación', 'lead')}: {ll} {tr('mes' if ll == 1 else 'meses', 'month' if ll == 1 else 'months')}</div></div>""",
                               unsafe_allow_html=True)
            st.write("")
        t1, t2 = st.container(key="row_tcal").columns([7, 1], vertical_alignment="center")
        t1.markdown(f'<div class="fc-ct">{tr("Calendario histórico", "History calendar")} · '
                    f'<span style="color:{YELLOW}">{tr("tocá un mes", "tap a month")}</span></div>', unsafe_allow_html=True)
        with t2.popover(tr("Info", "Info")):
            st.markdown(tr("Cada **fila es un año** y cada **columna un mes**. El color muestra cuánto fuego detectaron los "
                           "satélites: **azul oscuro = nada o poco**, **amarillo → rojo = mucho**.\n\n"
                           "En **«Comparado con lo normal»**: **rojo** = más fuego que en un año normal para ese mes, "
                           "**azul** = menos.\n\nTocá un cuadrito para ver el detalle y las noticias.",
                           "Each **row is a year**, each **column a month**. Color shows how much fire the satellites "
                           "detected: **dark blue = none/little**, **yellow → red = a lot**.\n\nIn **«Vs. normal»**: "
                           "**red** = more fire than a normal year for that month, **blue** = less.\n\nTap a square for details and news."))
        st.markdown(f"<div class='fc-tip'><b>{tr('Tocá cualquier cuadrito', 'Tap any square')}</b> "
                    f"{tr('para ver los números de ese mes, si fue normal y las noticias de ese momento.', 'to see that month’s numbers, whether it was normal, and the news from then.')}</div>",
                    unsafe_allow_html=True)
        if not HAS or not len(mt):
            st.info(tr("No hay focos en esta zona y período.", "No hotspots in this area and period."))
        else:
            years = sorted(mt.year.unique())  # eje x: años de izquierda a derecha; eje y: meses de abajo (ene) hacia arriba (dic)
            ylab = [str(y) for y in years]
            piv = lambda col: mt.pivot(index="year", columns="month", values=col).reindex(index=years, columns=range(1, 13))
            Z, MED, AN, PART, SRC = piv("focos"), piv("median"), piv("anom"), piv("partial"), piv("source")
            zcap = mt.loc[mt.source != "Terra", "focos"].max() if (mt.source != "Terra").any() else mt.focos.max()
            txt = np.empty(Z.shape, dtype=object)
            for i in range(Z.shape[0]):
                for j in range(12):
                    z = Z.values[i, j]
                    if np.isnan(z):
                        txt[i, j] = ""
                        continue
                    w, _ = anom_words(AN.values[i, j])
                    extra = ""
                    if PART.values[i, j] is True:
                        extra += "<br>" + tr("(mes incompleto)", "(incomplete month)")
                    if SRC.values[i, j] == "Terra":
                        extra += "<br>" + tr("(un solo satélite: menos confiable)", "(single satellite: less reliable)")
                    txt[i, j] = (f"<b>{MONTHS_L[j].capitalize()} {years[i]}</b><br>{fmt(z)} {tr('focos', 'hotspots')}<br>"
                                 f"{tr('Lo normal', 'Normal')}: {fmt(MED.values[i, j])}<br>{w}{extra}")
            if not ANOM:
                # escala raíz cuadrada: los meses sin fuego quedan azul oscuro y los picos en rojo
                base = mt.loc[mt.source != "Terra", "focos"] if (mt.source != "Terra").any() else mt.focos
                zcap = max(float(np.nanpercentile(base, 98)), 1.0)
                zv = np.sqrt(np.clip(Z.values, 0, zcap))
                nice = [0, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000, 50000]
                ok_t = [v for v in nice if v <= zcap * 0.6]
                ticks = ok_t[:: max(1, len(ok_t) // 3)][:4]
                hm = go.Heatmap(z=zv.T, x=ylab, y=MONTHS, colorscale=RAMPS[ramp], zmin=0, zmax=np.sqrt(zcap), xgap=2, ygap=2,
                                hoverinfo="skip",
                                colorbar=dict(title=tr("focos", "hotspots"), tickvals=[np.sqrt(v) for v in ticks] + [np.sqrt(zcap)],
                                              ticktext=[fmt(v) for v in ticks] + ["≥" + fmt(zcap)], orientation="h", y=-0.16,
                                              yanchor="top", thickness=10, len=0.8))
            else:
                lim = max(100, np.nanpercentile(np.abs(AN.values), 95) if np.isfinite(AN.values).any() else 100)
                hm = go.Heatmap(z=np.clip(AN.values, -100, lim).T, x=ylab, y=MONTHS, colorscale=ANOM_RAMP, zmin=-100,
                                zmax=lim, xgap=2, ygap=2, hoverinfo="skip",
                                colorbar=dict(title=tr("vs. lo normal (%)", "vs. normal (%)"), orientation="h", y=-0.16,
                                              yanchor="top", thickness=10, len=0.7, ticksuffix=" %"))
            # capa invisible para poder tocar los cuadritos
            gx, gy = np.meshgrid(ylab, MONTHS)
            ok = ~np.isnan(Z.values.T)
            hit = go.Scatter(x=gx[ok], y=gy[ok], mode="markers", text=txt.T[ok], hovertemplate="%{text}<extra></extra>",
                             marker=dict(symbol="square", size=18, color="rgba(0,0,0,0)", opacity=0.01),
                             selected=dict(marker=dict(opacity=0.01)), unselected=dict(marker=dict(opacity=0.01)),
                             showlegend=False)
            fig = go.Figure([hm, hit])
            fig.update_yaxes(type="category", categoryorder="array", categoryarray=MONTHS, fixedrange=True,
                             tickfont=dict(size=11))
            xt = [str(y) for y in years if y % 5 == 0 or (y == years[0] and y % 5 <= 2) or (y == years[-1] and y % 5 >= 3)]
            fig.update_xaxes(side="bottom", fixedrange=True, type="category", categoryorder="array", categoryarray=ylab,
                             tickangle=0, tickvals=xt, tickfont=dict(size=11))
            style_fig(fig, 500, clickmode="event+select", dragmode=False,
                      margin=dict(l=6, r=6, t=10, b=70), hoverlabel=dict(bgcolor=NAVY, bordercolor=YELLOW))
            ev = st.plotly_chart(fig, width="stretch", on_select="rerun", selection_mode="points",
                                 key=f"cal_{st.session_state.map_v}",
                                 config={"displayModeBar": False, "toImageButtonOptions": {"filename": "firechron_calendario"}})
            pts = (ev or {}).get("selection", {}).get("points", []) if ev else []
            if pts:
                p0 = pts[0]
                try:
                    sel = (int(p0["x"]), MONTHS.index(p0["y"]) + 1)
                except (KeyError, ValueError):
                    sel = None
                if sel and st.session_state.get("cal_sel") != sel:
                    st.session_state.cal_sel = sel
                    month_detail(*sel)
            else:
                st.session_state.cal_sel = None
            with st.expander(tr("¿No podés tocar el calendario? Elegí el mes acá", "Can't tap? Pick the month here")):
                opts = [(int(r.year), int(r.month)) for r in mt.sort_values(["year", "month"], ascending=False).itertuples()]
                pk = st.selectbox(tr("Mes", "Month"), opts, format_func=lambda t: f"{MONTHS_L[t[1] - 1].capitalize()} {t[0]}")
                if st.button(tr("Ver detalle y noticias", "See details and news")):
                    month_detail(*pk)

# ================================================================== 4. MÁS DETALLE
st.write("")
st.markdown(f"### {tr('Explorá más', 'Explore more')}")
tabs = st.tabs([tr("Este mes vs. otros años", "This month vs. other years"), tr("Temporada típica", "Typical season"),
                tr("Año por año", "Year by year"), tr("Cómo funciona", "How it works"),
                tr("Reportá un fuego", "Report a fire"), tr("Descargas", "Downloads"),
                tr("Fuentes y créditos", "Sources and credits")])

if HAS:
    with tabs[0]:
        opts = [(y, m) for y in range(DMAX.year, REF_START - 1, -1) for m in range(12, 0, -1) if dt.date(y, m, 1) <= DMAX]
        y, m = st.selectbox(tr("Mes a analizar", "Month"), opts, format_func=lambda t: f"{MONTHS_L[t[1] - 1].capitalize()} {t[0]}")
        C = compare_month(daily, y, m, DMAX, REF)
        aw, acls = anom_words(C["anom"])
        st.markdown(tr(f"Del 1 al {C['D']} de {MONTHS_L[m - 1]} de {y} hubo **{fmt(C['cur'])} focos**; en un año normal hay "
                       f"**{fmt(C['med'])}** → <span class='{acls}'><b>{aw}</b></span>. Puesto {C['rank']} de {C['n']} años.",
                       f"From 1 to {C['D']} {MONTHS_L[m - 1]} {y}: **{fmt(C['cur'])} hotspots**; a normal year has "
                       f"**{fmt(C['med'])}** → <span class='{acls}'><b>{aw}</b></span>. Rank {C['rank']} of {C['n']} years."),
                    unsafe_allow_html=True)
        mlen = pd.Period(f"{y}-{m:02d}").days_in_month
        cums = np.array([daily[(daily.year == yy) & (daily.month == m)].set_index("day").focos
                         .reindex(range(1, mlen + 1)).fillna(0).cumsum().values for yy in range(REF[0], REF[1] + 1) if yy != y])
        dd = np.arange(1, mlen + 1)
        q25, q50, q75 = np.percentile(cums, [25, 50, 75], axis=0)
        cur = daily[(daily.year == y) & (daily.month == m) & (daily.day <= C["D"])].set_index("day").focos.cumsum()
        f2 = go.Figure()
        f2.add_trace(go.Scatter(x=np.r_[dd, dd[::-1]], y=np.r_[q75, q25[::-1]], fill="toself", line=dict(width=0),
                                fillcolor="rgba(46,150,245,0.25)", name=tr("Rango normal", "Normal range"), hoverinfo="skip"))
        f2.add_trace(go.Scatter(x=dd, y=q50, line=dict(color=WHITE, dash="dash"), name=tr("Lo normal", "Normal")))
        f2.add_trace(go.Scatter(x=cur.index, y=cur.values, line=dict(color=YELLOW, width=3), name=f"{MONTHS[m - 1]} {y}"))
        st.plotly_chart(style_fig(f2, 380, title=tr("Focos acumulados día a día", "Cumulative hotspots day by day"),
                                  xaxis_title=tr("Día del mes", "Day of month"), hovermode="x unified",
                                  legend=dict(orientation="h", y=-0.2)), width="stretch", config={"displayModeBar": False})

    with tabs[1]:
        ref_m = full[(full.year >= REF[0]) & (full.year <= REF[1])]
        qq = ref_m.groupby("month").focos.quantile([0.25, 0.5, 0.75]).unstack()
        hy = st.selectbox(tr("Comparar con el año", "Compare with year"), sorted(full.year.unique(), reverse=True))
        f4 = go.Figure()
        f4.add_trace(go.Scatter(x=MONTHS + MONTHS[::-1], y=np.r_[qq[0.75].values, qq[0.25].values[::-1]], fill="toself",
                                fillcolor="rgba(228,55,0,0.25)", line=dict(width=0), name=tr("Rango normal", "Normal range"), hoverinfo="skip"))
        f4.add_trace(go.Scatter(x=MONTHS, y=qq[0.5].values, line=dict(color=RED, width=3), name=tr("Mes normal", "Normal month")))
        hs = full[full.year == hy].set_index("month").focos.reindex(range(1, 13))
        f4.add_trace(go.Scatter(x=MONTHS, y=hs.values, line=dict(color=YELLOW, width=2), mode="lines+markers", name=str(hy)))
        st.plotly_chart(style_fig(f4, 400, title=tr("¿En qué meses se quema esta zona?", "When does this area burn?"),
                                  hovermode="x unified", legend=dict(orientation="h", y=-0.15)), width="stretch", config={"displayModeBar": False})

    with tabs[2]:
        ya = mt.groupby("year").agg(focos=("focos", "sum"), ndays=("ndays", "sum")).reset_index()
        ya["full"] = [nd >= (366 if pd.Timestamp(f"{yy}-12-31").is_leap_year else 365) for yy, nd in zip(ya.year, ya.ndays)]
        f5 = go.Figure(go.Bar(x=ya.year, y=ya.focos, marker_color=[BLUE_L if f else "#5E6C8F" for f in ya.full],
                              customdata=np.where(ya.full, "", tr(" (año incompleto)", " (incomplete year)")),
                              hovertemplate="%{x}: %{y:,.0f}%{customdata}<extra></extra>"))
        f5.add_hline(y=ya[ya.full].focos.median(), line_dash="dash", line_color=WHITE, annotation_text=tr("año normal", "normal year"))
        st.plotly_chart(style_fig(f5, 400, title=tr("Total de focos por año", "Total hotspots per year"), showlegend=False), width="stretch", config={"displayModeBar": False})
        st.caption(tr("Las barras grises son años incompletos (2000 empieza en noviembre; 2026 llega hasta "
                      f"{DMAX:%d/%m}).", f"Grey bars are incomplete years (2000 starts in November; 2026 runs to {DMAX:%m/%d})."))

    with tabs[3]:
        st.markdown(tr(
            "**El problema.** La NASA mira el fuego desde el espacio con dos tipos de satélites: **MODIS** (desde 2000, "
            "ve cuadros de 1 km) y **VIIRS** (desde 2012, ve cuadros de 375 m: más fino, detecta más fuegos chicos). "
            "Si se juntan sin más, parece que en 2012 «explotaron» los incendios. Es un efecto del cambio de satélite.\n\n"
            "**La solución: armonizar.** Contamos el fuego de la misma forma para los dos (cuadros de 1 km por día) y, "
            "durante los años en que funcionaron juntos (2012–2017), medimos cuánto más ve VIIRS en cada región. "
            "Con ese factor «traducimos» MODIS a la escala de VIIRS. Lo comprobamos en 2018–2020: el error total es menor al 2 %.\n\n"
            "**El pronóstico.** Un modelo de aprendizaje automático (LightGBM) aprendió de 2003–2019 cómo la historia "
            "reciente del fuego, el clima y El Niño anticipan una temporada más activa de lo normal.",
            "**The problem.** NASA watches fire from space with two kinds of satellites: **MODIS** (since 2000, 1 km "
            "pixels) and **VIIRS** (since 2012, 375 m pixels: finer, detects more small fires). Simply merged, fires seem "
            "to «explode» in 2012 — an artifact of the satellite change.\n\n**The fix: harmonize.** We count fire the same "
            "way for both (1-km cells per day) and, during the years both flew (2012–2017), measure how much more VIIRS "
            "sees in each region. That factor «translates» MODIS to the VIIRS scale. Checked on 2018–2020: total error "
            "below 2%.\n\n**The outlook.** A machine-learning model (LightGBM) learned from 2003–2019 how recent fire "
            "history, climate and El Niño anticipate an above-normal season."))
        yraw = daily.groupby("year").agg(focos=("focos", "sum"), crudo=("crudo", "sum")).reset_index()
        yraw = yraw[(yraw.year >= 2003) & (yraw.year < DMAX.year)]
        f7 = go.Figure()
        f7.add_trace(go.Scatter(x=yraw.year, y=yraw.crudo, mode="lines+markers", line=dict(color="#8A94AD", dash="dot"),
                                name=tr("Sin armonizar (salto falso en 2012)", "Not harmonized (false jump in 2012)")))
        f7.add_trace(go.Scatter(x=yraw.year, y=yraw.focos, mode="lines+markers", line=dict(color=YELLOW, width=3),
                                name=tr("Firechron (armonizado)", "Firechron (harmonized)")))
        f7.add_vline(x=2012, line_dash="dot", line_color=WHITE, annotation_text="MODIS → VIIRS")
        st.plotly_chart(style_fig(f7, 360, title=tr("Antes y después de armonizar (esta zona)", "Before and after harmonizing (this area)"),
                                  legend=dict(orientation="h", y=-0.2)), width="stretch", config={"displayModeBar": False})

with tabs[4]:
    st.markdown(tr("¿Viste humo o fuego? **Contanos dónde y cuándo.** Firechron cruza tu reporte con lo que detectaron "
                   "los satélites cerca de ese lugar y te dice cuándo vuelve a pasar el próximo. "
                   "*Prototipo: no reemplaza al 911 ni a los bomberos.*",
                   "Saw smoke or fire? **Tell us where and when.** Firechron cross-checks your report with what satellites "
                   "detected nearby and tells you when the next one passes. *Prototype: does not replace emergency services.*"))
    cen = GEOM.representative_point()
    with st.form("reporte"):
        c1, c2, c3 = st.columns(3)
        qu = c1.selectbox(tr("¿Qué viste?", "What did you see?"), [tr("Humo", "Smoke"), tr("Llamas", "Flames"), tr("Zona quemada", "Burned area")])
        fday = c2.date_input(tr("Fecha", "Date"), value=dt.date.today(), max_value=dt.date.today(), format="DD/MM/YYYY")
        hora = c3.time_input(tr("Hora aprox.", "Approx. time"), value=dt.time(15, 0))
        c4, c5 = st.columns(2)
        rlat = c4.number_input("Lat.", -60.0, 15.0, round(cen.y, 3), 0.01, format="%.3f")
        rlon = c5.number_input("Lon.", -85.0, -30.0, round(cen.x, 3), 0.01, format="%.3f")
        com = st.text_input(tr("Comentario (opcional, sin datos personales)", "Comment (optional, no personal data)"), max_chars=200)
        sent = st.form_submit_button(tr("Enviar reporte", "Send report"), type="primary")
    if sent:
        try:
            new = not os.path.exists(REPORTES)
            with open(REPORTES, "a", newline="", encoding="utf-8") as fh:
                w = csv.writer(fh)
                if new:
                    w.writerow(["enviado_utc", "fecha", "hora", "tipo", "lat", "lon", "comentario"])
                w.writerow([dt.datetime.utcnow().isoformat(timespec="seconds"), fday, hora, qu, rlat, rlon, com])
        except OSError:
            pass
        st.success(tr("¡Gracias! Reporte recibido.", "Thanks! Report received."))
        # 1) archivo histórico de la app (±1 día, ±0,1°)
        if fday <= DMAX:
            a_, b_ = np.datetime64(fday - dt.timedelta(days=1), "D").astype(np.int64), np.datetime64(fday + dt.timedelta(days=1), "D").astype(np.int64)
            gr, gc = math.floor(rlat / GRID), math.floor(rlon / GRID)
            m_ = ((np.abs(data["g_r"] - gr) <= 1) & (np.abs(data["g_c"] - gc) <= 1) & (data["days"] >= a_) & (data["days"] <= b_))
            n = float(data["focos"][m_].sum())
            st.markdown(tr(f"🛰️ **Registro satelital:** {fmt(n)} focos a menos de ~15 km entre el día anterior y el siguiente." if n > 0
                           else "🛰️ **Registro satelital:** sin focos a menos de ~15 km en esos días (puede ser un fuego chico, nubes o humo sin llama).",
                           f"🛰️ **Satellite record:** {fmt(n)} hotspots within ~15 km from the day before to the day after." if n > 0
                           else "🛰️ **Satellite record:** no hotspots within ~15 km on those days (small fire, clouds, or smoke only)."))
        # 2) FIRMS en tiempo casi real (si hay clave configurada)
        key = None
        try:
            key = st.secrets.get("FIRMS_MAP_KEY")
        except Exception:  # noqa: BLE001
            key = os.environ.get("FIRMS_MAP_KEY")
        if key and fday > DMAX - dt.timedelta(days=9):
            tot = 0
            for srcx in ["VIIRS_NOAA20_NRT", "VIIRS_NOAA21_NRT", "MODIS_NRT"]:
                try:
                    url = (f"https://firms.modaps.eosdis.nasa.gov/api/area/csv/{key}/{srcx}/"
                           f"{rlon - .15:.3f},{rlat - .15:.3f},{rlon + .15:.3f},{rlat + .15:.3f}/2/{fday:%Y-%m-%d}")
                    r = requests.get(url, timeout=15)
                    tot += max(0, len(r.text.strip().splitlines()) - 1)
                except Exception:  # noqa: BLE001
                    pass
            st.markdown(tr(f"🔥 **NASA FIRMS (tiempo casi real):** {tot} detecciones cerca en las últimas 48 h.",
                           f"🔥 **NASA FIRMS (near real time):** {tot} detections nearby in the last 48 h."))
        lt = 13.5 - rlon / 15 - 3  # hora local (UTC−3) de la pasada diurna de VIIRS
        st.info(tr(f"⏱️ Los satélites VIIRS (NOAA-20 y NOAA-21) pasan por ahí cerca de las **{lt:.0f} h** y de la "
                   f"**{(lt + 12) % 24:.0f} h** (hora argentina, ±1,5 h). Sus detecciones se publican en NASA FIRMS en ~3 horas.",
                   f"⏱️ VIIRS satellites (NOAA-20, NOAA-21) pass nearby around **{lt:.0f}:00** and **{(lt + 12) % 24:.0f}:00** "
                   "(UTC−3, ±1.5 h). Detections appear on NASA FIRMS within ~3 hours."))

with tabs[5]:
    if HAS:
        st.caption(tr("Cada descarga es un ZIP con el CSV y un LEEME con unidades, método y la cita de NASA FIRMS.",
                      "Each download is a ZIP with the CSV and a README with units, method and the NASA FIRMS citation."))
        tag = f"{D0:%Y%m%d}_{D1:%Y%m%d}"
        units = "harmonized hotspots = 1-km cells with fire per day, VIIRS S-NPP scale"
        desc_zone = f"Area: {st.session_state.zone['label']}\nWKT (EPSG:4326): {WKT}\nPeriod: {D0} to {D1}"
        out_m = mt[["year", "month", "focos", "median", "anom", "partial"]].rename(columns={
            "focos": "focos_armonizados", "median": f"mediana_{REF[0]}_{REF[1]}", "anom": "anomalia_pct",
            "partial": "mes_incompleto"})
        st.download_button(tr("Calendario mensual (CSV)", "Monthly calendar (CSV)"),
                           zip_bytes(f"firechron_mensual_{tag}", out_m.round(2), desc_zone, units),
                           file_name=f"firechron_mensual_{tag}.zip", mime="application/zip", type="primary")
        if st.button(tr("Preparar focos diarios por cuadro de ~11 km", "Prepare daily hotspots per ~11 km cell"), key="p_daily"):
            c = zone_cells(WKT, D0, D1)
            dfc = pd.DataFrame({"fecha": c.days.values.astype("datetime64[D]"), "lat": ((c.g_r + 0.5) * GRID).round(2),
                                "lon": ((c.g_c + 0.5) * GRID).round(2), "focos_armonizados": c.focos.round(3)}
                               ).sort_values(["fecha", "lat", "lon"])
            st.session_state["z_daily"] = zip_bytes(f"firechron_diario_0p1_{tag}", dfc, desc_zone, units)
        if "z_daily" in st.session_state:
            st.download_button(tr("Focos diarios (CSV)", "Daily hotspots (CSV)"), st.session_state["z_daily"],
                               file_name=f"firechron_diario_0p1_{tag}.zip", mime="application/zip", key="d_daily")
        if pz_ok := (FORECAST and len(pz)):
            st.download_button(tr("Pronóstico de la zona (CSV)", "Area outlook (CSV)"),
                               zip_bytes(f"firechron_pronostico_{FY}{FM:02d}",
                                         pz[["lat", "lon", "year", "month", "lead", "prob", "p_clim"]].round(4), desc_zone,
                                         "prob = probability that the month has more hotspots than the 2003-2019 median for that month and 0.5° cell"),
                               file_name=f"firechron_pronostico_{FY}{FM:02d}.zip", mime="application/zip")
        minx, miny, maxx, maxy = GEOM.bounds
        if os.path.exists(DATA_1KM) and st.button(tr("Preparar focos armonizados de 1 km", "Prepare 1-km harmonized hotspots"), key="p_1km"):
            qd = duckdb.sql(f"""SELECT acq_date AS fecha, CAST(lat AS DOUBLE) lat, CAST(lon AS DOUBLE) lon, fuente,
                                round(peso, 4) AS focos_armonizados FROM read_parquet('{DATA_1KM}')
                                WHERE lat BETWEEN {miny} AND {maxy} AND lon BETWEEN {minx} AND {maxx}
                                  AND acq_date BETWEEN '{D0}' AND '{D1}'""").df()
            qd = qd[shapely.contains_xy(GEOM, qd.lon.values, qd.lat.values)] if len(qd) else qd
            if len(qd) <= MAX_RAW_ROWS:
                st.session_state["z_1km"] = zip_bytes(f"firechron_1km_{tag}", qd.sort_values("fecha"), desc_zone, units)
            else:
                st.warning(tr("Demasiados datos: achicá la zona o el período.", "Too much data: reduce area or period."))
        if "z_1km" in st.session_state:
            st.download_button("1 km (CSV)", st.session_state["z_1km"], file_name=f"firechron_1km_{tag}.zip",
                               mime="application/zip", key="d_1km")

with tabs[6]:
    st.markdown(tr("""
**Datos de la NASA (núcleo del proyecto)**
- **NASA FIRMS** — Fire Information for Resource Management System: focos de calor MODIS Collection 6.1 (Terra y Aqua, 1 km; MCD14ML) y VIIRS 375 m (Suomi NPP — VNP14IMG — y NOAA-20 — VJ114IMG), archivo y tiempo casi real. https://firms.modaps.eosdis.nasa.gov
- *We acknowledge the use of data and/or imagery from NASA's Fire Information for Resource Management System (FIRMS) (https://earthdata.nasa.gov/firms), part of NASA's Land, Atmosphere Near real-time Capability for Earth observations (LANCE) (https://earthdata.nasa.gov/lance) and NASA's Earth Science Data and Information System (ESDIS).*

**Otros datos abiertos**
- **ERA5-Land monthly** (Copernicus Climate Change Service, ECMWF; Muñoz-Sabater et al., 2021) — licencia CC-BY 4.0. *Contains modified Copernicus Climate Change Service information 2026.*
- **Índice ONI** (El Niño/La Niña) — NOAA Climate Prediction Center, dominio público.
- **Natural Earth** admin-1 (límites de provincias) — dominio público.
- **OpenStreetMap / Nominatim** (buscador de lugares) — © colaboradores de OpenStreetMap, ODbL. Mapa base © OpenStreetMap.
- **Google News** (feed RSS público) y **GDELT Project** para titulares; **Wikipedia** (CC BY-SA). Sólo se muestran titulares y enlaces.

**Software libre:** Python, pandas, NumPy, PyArrow, DuckDB, Shapely, Streamlit, streamlit-folium, Folium/Leaflet, Leaflet.draw, Plotly, LightGBM, PyTorch, scikit-learn, xarray. Tipografía Fira Sans (SIL OFL). Colores de la guía de marca de NASA Space Apps.

**Uso de inteligencia artificial:** usamos **Claude (Anthropic)** como asistente para programar, documentar y redactar textos. Todo el código se corrió y los resultados se revisaron por el equipo.

**Licencia:** código MIT. Firechron es un proyecto del equipo **BurnInHell** para el NASA Space Apps Challenge 2026 (Mendoza). **No es una herramienta oficial de NASA** ni está avalado por la NASA.
""", """
**NASA data (project core)**
- **NASA FIRMS** — Fire Information for Resource Management System: MODIS Collection 6.1 hotspots (Terra and Aqua, 1 km; MCD14ML) and VIIRS 375 m (Suomi NPP — VNP14IMG — and NOAA-20 — VJ114IMG), archive and near real time. https://firms.modaps.eosdis.nasa.gov
- *We acknowledge the use of data and/or imagery from NASA's Fire Information for Resource Management System (FIRMS) (https://earthdata.nasa.gov/firms), part of NASA's Land, Atmosphere Near real-time Capability for Earth observations (LANCE) (https://earthdata.nasa.gov/lance) and NASA's Earth Science Data and Information System (ESDIS).*

**Other open data**
- **ERA5-Land monthly** (Copernicus Climate Change Service, ECMWF; Muñoz-Sabater et al., 2021) — CC-BY 4.0. *Contains modified Copernicus Climate Change Service information 2026.*
- **ONI index** (El Niño/La Niña) — NOAA Climate Prediction Center, public domain.
- **Natural Earth** admin-1 boundaries — public domain.
- **OpenStreetMap / Nominatim** (place search) — © OpenStreetMap contributors, ODbL. Basemap © OpenStreetMap.
- **Google News** (public RSS feed) and **GDELT Project** for headlines; **Wikipedia** (CC BY-SA). Only headlines and links are shown.

**Open-source software:** Python, pandas, NumPy, PyArrow, DuckDB, Shapely, Streamlit, streamlit-folium, Folium/Leaflet, Leaflet.draw, Plotly, LightGBM, PyTorch, scikit-learn, xarray. Fira Sans typeface (SIL OFL). Colors from the NASA Space Apps brand guide.

**Use of AI:** we used **Claude (Anthropic)** as an assistant for coding, documentation and copywriting. All code was run and all results reviewed by the team.

**License:** MIT code. Firechron is a project by team **BurnInHell** for the NASA Space Apps Challenge 2026 (Mendoza). **Not an official NASA tool** nor endorsed by NASA.
"""))

if st.session_state.pop("zone_toast", None) and "toasted_once" in st.session_state:
    st.toast(tr(f"Zona actualizada: {st.session_state.zone['label']}. Bajá para ver el calendario.",
                f"Area updated: {st.session_state.zone['label']}. Scroll down for the calendar."))
st.session_state.toasted_once = True
st.markdown(f"<div class='fc-foot'>Firechron · {tr('Equipo', 'Team')} BurnInHell · NASA Space Apps Challenge 2026 · "
            f"{tr('Datos', 'Data')}: NASA FIRMS (MODIS + VIIRS) · {tr('No es una herramienta oficial de NASA.', 'Not an official NASA tool.')}"
            f"<br>{ACK}</div>", unsafe_allow_html=True)
