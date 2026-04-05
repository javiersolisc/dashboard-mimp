import os, json
import pandas as pd
import numpy as np
import geopandas as gpd
import plotly.express as px
import plotly.graph_objects as go
import dash
from dash import dcc, html, Input, Output, State, dash_table, Patch
import dash_bootstrap_components as dbc

# ── PATHS ────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SHP_PATH = os.path.join(BASE_DIR, "Distrito_INEI_2017", "Distrito_INEI_2017.shp")
CSV_PATH = os.path.join(BASE_DIR, "data", "datos_distritos.csv")

# ── PALETTE ──────────────────────────────────────────────────────────────────
C = {
    "purple_dark":  "#26215c",
    "purple_mid":   "#3c3489",
    "purple_light": "#534ab7",
    "purple_pale":  "#eeedfe",
    "green_dark":   "#0f6e56",
    "green_mid":    "#1d9e75",
    "green_light":  "#5dcaa5",
    "green_pale":   "#e1f5ee",
    "orange_dark":  "#854f0b",
    "orange_mid":   "#ef9f27",
    "orange_light": "#f2b15a",
    "orange_pale":  "#faeeda",
    "gray_dark":    "#2c2c2a",
    "gray_mid":     "#5f5e5a",
    "gray_light":   "#b4b2a9",
    "gray_pale":    "#f1efe8",
    "bg":           "#f8f7f4",
    "white":        "#ffffff",
    "red_risk":     "#c0392b",
    "sem_critico":  "#c0392b",
    "sem_medio":    "#ef9f27",
    "sem_adecuado": "#1d9e75",
}
SEM_COLOR = {"Crítico": C["sem_critico"], "Medio": C["sem_medio"], "Adecuado": C["sem_adecuado"]}

# ── LOAD & PREPARE SHAPEFILE ─────────────────────────────────────────────────
print("⏳ Cargando shapefile...")
gdf_raw = gpd.read_file(SHP_PATH)
gdf_raw = gdf_raw.to_crs(epsg=4326)

# Normalize column names (lowercase)
gdf_raw.columns = [c.lower() for c in gdf_raw.columns]

# Ensure ubigeo is 6-char zero-padded string
gdf_raw["ubigeo"] = gdf_raw["ubigeo"].astype(str).str.zfill(6)

# Simplify geometry for browser performance (tolerance in degrees ~1km)
print("⏳ Simplificando geometrías...")
gdf_raw["geometry"] = gdf_raw["geometry"].simplify(0.005, preserve_topology=True)

# Build GeoJSON with ubigeo as feature id
geojson_dict = json.loads(gdf_raw.to_json())
for feat in geojson_dict["features"]:
    feat["id"] = feat["properties"]["ubigeo"]

print(f"✅ Shapefile cargado: {len(gdf_raw)} distritos")

# ── CENTROIDS FOR LABELS ─────────────────────────────────────────────────────
# Project to Peru's UTM zone (EPSG:32718) for accurate centroids, then back to WGS84
gdf_proj = gdf_raw.to_crs(epsg=32718)

def _centroids_wgs84(gdf_p, dissolve_by=None):
    """Dissolve (optional), compute centroid in projected CRS, return lon/lat in WGS84."""
    if dissolve_by:
        g = gdf_p.dissolve(by=dissolve_by).reset_index()
    else:
        g = gdf_p.copy()
    cents = g.copy()
    cents["geometry"] = g.geometry.centroid          # centroid in projected CRS
    cents = cents.to_crs(epsg=4326)                  # back to WGS84
    cents["lon"] = cents.geometry.x
    cents["lat"] = cents.geometry.y
    return cents

dept_cent = _centroids_wgs84(gdf_proj, dissolve_by="nombdep")
prov_cent = _centroids_wgs84(gdf_proj, dissolve_by=["nombdep","nombprov"])
dist_cent = _centroids_wgs84(gdf_proj)               # one row per district

# ── GENERATE / LOAD SYNTHETIC DATA ───────────────────────────────────────────
def generate_data(gdf: gpd.GeoDataFrame) -> pd.DataFrame:
    """Generate synthetic risk/access scores keyed by real ubigeo."""
    np.random.seed(42)
    n = len(gdf)
    df = pd.DataFrame({
        "ubigeo":    gdf["ubigeo"].values,
        "district":  gdf["nombdist"].str.title().values,
        "provincia": gdf["nombprov"].str.title().values,
        "region":    gdf["nombdep"].str.title().values,
    })

    # Rural flag (approx: small/remote depts tend more rural)
    rural_depts = {"Amazonas","Apurímac","Ayacucho","Huancavelica","Pasco",
                   "Madre De Dios","Ucayali","Loreto","San Martín","Huánuco"}
    df["area"] = np.where(
        df["region"].isin(rural_depts),
        np.random.choice(["Urbana","Rural"], n, p=[0.35, 0.65]),
        np.random.choice(["Urbana","Rural"], n, p=[0.65, 0.35])
    )

    df["pobreza_pct"] = np.where(
        df["area"] == "Rural",
        np.clip(np.random.beta(3,4,n)*80+20, 5, 98),
        np.clip(np.random.beta(2,6,n)*60+5,  2, 85)
    ).round(1)
    df["pobreza_quintil"] = pd.cut(df["pobreza_pct"],
        bins=[0,20,40,60,80,100], labels=["Q1","Q2","Q3","Q4","Q5"])

    df["educ_baja_pct"] = np.where(
        df["area"] == "Rural",
        np.clip(np.random.beta(4,3,n)*70+15, 5, 98),
        np.clip(np.random.beta(2,5,n)*50+5,  2, 80)
    ).round(1)
    df["nivel_educativo"] = pd.cut(df["educ_baja_pct"],
        bins=[0,25,50,75,100], labels=["Alto","Medio","Bajo","Muy bajo"])

    df["pct_mujeres_18_35"] = (np.random.beta(3,3,n)*40+20).round(1)
    df["poblacion_mujeres"]  = np.where(
        df["area"] == "Rural",
        np.random.randint(200, 6000, n),
        np.random.randint(1500, 90000, n)
    ).astype(int)
    df["tolerancia_social"] = np.random.beta(3,4,n).round(3)

    # Risk score
    base_risk = (
        0.30*(df["pobreza_pct"]/100) +
        0.25*(df["educ_baja_pct"]/100) +
        0.20*(df["area"]=="Rural").astype(float) +
        0.15*df["tolerancia_social"] +
        0.10*(df["pct_mujeres_18_35"]/100)*2
    )
    df["riesgo"] = np.clip(base_risk + np.random.normal(0,0.05,n), 0.05, 0.95).round(3)

    # Services
    df["n_cem"]            = np.random.choice([0,1,2,3], n, p=[0.55,0.30,0.10,0.05])
    df["n_refugios"]       = np.random.choice([0,1,2],   n, p=[0.75,0.20,0.05])
    df["n_otros_servicios"]= np.random.choice([0,1,2,3,4],n,p=[0.30,0.30,0.20,0.15,0.05])
    df["linea100_cobertura"] = np.random.choice([True,False], n, p=[0.65,0.35])
    df["acceso_telefonia_pct"] = np.where(
        df["area"]=="Rural",
        np.clip(np.random.beta(3,5,n)*70+10, 5, 95),
        np.clip(np.random.beta(5,3,n)*30+60, 30, 98)
    ).round(1)

    # Access score
    total_svc = df["n_cem"] + df["n_refugios"] + df["n_otros_servicios"]
    df["acceso"] = np.clip(
        (total_svc/7)*0.6 +
        (df["acceso_telefonia_pct"]/100)*0.25 +
        df["linea100_cobertura"].astype(float)*0.15 +
        np.random.normal(0,0.03,n),
        0.02, 0.95
    ).round(3)

    df["prioridad"] = (df["riesgo"] * (1 - df["acceso"])).round(3)
    df["riesgo_cat"] = pd.cut(df["riesgo"],
        bins=[0,0.33,0.55,0.75,1.0], labels=["Bajo","Medio","Alto","Crítico"])
    df["semaforo"] = np.where(df["prioridad"]>0.50,"Crítico",
                     np.where(df["prioridad"]>0.30,"Medio","Adecuado"))

    def rec(r):
        if r["riesgo"] > 0.6 and r["n_cem"] == 0:
            return "Alto riesgo sin CEM → Priorizar implementación de CEM"
        elif r["riesgo"] > 0.6 and r["acceso"] < 0.4:
            return "Alto riesgo con acceso bajo → Fortalecer servicios existentes"
        elif r["riesgo"] > 0.4 and r["n_cem"] == 0:
            return "Riesgo medio sin CEM → Considerar punto de atención"
        elif r["tolerancia_social"] > 0.6:
            return "Alta tolerancia social → Priorizar campañas de sensibilización"
        else:
            return "Mantener monitoreo y reforzar capacidades locales"

    df["recomendacion"] = df.apply(rec, axis=1)
    return df

# ── LOAD OR GENERATE DATA ────────────────────────────────────────────────────
# Strategy:
#   1. If CSV exists AND has 'ubigeo' column AND row count matches shapefile → use it as-is
#   2. If CSV exists but is missing 'ubigeo' (older format) → merge by district name,
#      add ubigeo from shapefile, fill gaps with synthetic scores, save updated CSV
#   3. If CSV doesn't exist → generate fully synthetic data from shapefile, save it
#
# This also handles the case where new districts appear or old ones disappear:
# the shapefile is always the authoritative geometry source; any district in the
# shapefile but missing from the CSV gets synthetic scores assigned automatically.

def _merge_csv_with_shapefile(df_csv: pd.DataFrame, gdf: gpd.GeoDataFrame) -> pd.DataFrame:
    """
    Join an existing CSV (without ubigeo) to the shapefile by district + province + dept name.
    Districts in the shapefile but absent from the CSV receive synthetic scores.
    Districts in the CSV but absent from the shapefile are silently dropped
    (they no longer exist or were renamed).
    """
    # Build a lookup key from the shapefile (title-cased, stripped)
    shp = pd.DataFrame({
        "ubigeo":    gdf["ubigeo"].values,
        "district":  gdf["nombdist"].str.strip().str.title().values,
        "provincia": gdf["nombprov"].str.strip().str.title().values,
        "region":    gdf["nombdep"].str.strip().str.title().values,
    })

    # Normalise CSV names for matching
    for col in ["district","provincia","region"]:
        if col in df_csv.columns:
            df_csv[col] = df_csv[col].astype(str).str.strip().str.title()

    # Try to join on all three levels; fall back to district+region if provincia missing
    join_cols = [c for c in ["district","provincia","region"] if c in df_csv.columns]
    merged = shp.merge(df_csv, on=join_cols, how="left")

    # Rows that didn't match get synthetic scores (same generator, different seed offset)
    missing_mask = merged["riesgo"].isna()
    n_missing = missing_mask.sum()
    if n_missing:
        print(f"  ℹ️  {n_missing} distritos sin datos en CSV → scores sintéticos asignados")
        np.random.seed(99)
        synthetic = generate_data(gdf.iloc[merged[missing_mask].index].reset_index(drop=True))
        for col in synthetic.columns:
            if col not in ("ubigeo","district","provincia","region") and col in merged.columns:
                merged.loc[missing_mask, col] = synthetic[col].values

    return merged

# ── actual load logic ─────────────────────────────────────────────────────────
if os.path.exists(CSV_PATH):
    df_csv = pd.read_csv(CSV_PATH)
    has_ubigeo = "ubigeo" in df_csv.columns
    row_match  = has_ubigeo and len(df_csv) == len(gdf_raw)

    if row_match:
        # Happy path: CSV already linked to shapefile ubigeos
        df_orig = df_csv
        df_orig["ubigeo"] = df_orig["ubigeo"].astype(str).str.zfill(6)
        print(f"✅ CSV cargado directamente ({len(df_orig)} filas)")
    else:
        # CSV exists but lacks ubigeo or has different district count
        reason = "sin columna ubigeo" if not has_ubigeo else f"filas CSV={len(df_csv)} vs shapefile={len(gdf_raw)}"
        print(f"⏳ CSV encontrado pero {reason} → uniendo con shapefile por nombre...")
        df_orig = _merge_csv_with_shapefile(df_csv, gdf_raw)
        df_orig.to_csv(CSV_PATH, index=False)
        print(f"✅ CSV actualizado y guardado ({len(df_orig)} filas)")
else:
    print("⏳ CSV no encontrado → generando datos sintéticos desde shapefile...")
    df_orig = generate_data(gdf_raw)
    os.makedirs(os.path.dirname(CSV_PATH), exist_ok=True)
    df_orig.to_csv(CSV_PATH, index=False)
    print(f"✅ CSV generado ({len(df_orig)} filas)")

print(f"✅ Datos: {len(df_orig)} filas")

# ── FILTER OPTIONS ───────────────────────────────────────────────────────────
REGIONS   = sorted(df_orig["region"].unique())
PROVINCIAS= sorted(df_orig["provincia"].unique())
AREAS     = ["Urbana", "Rural"]
EDUC      = ["Alto", "Medio", "Bajo", "Muy bajo"]
QUINTILES = ["Q1", "Q2", "Q3", "Q4", "Q5"]

# ── HELPERS ──────────────────────────────────────────────────────────────────
def filter_df(region, prov, area, educ, quintil):
    d = df_orig.copy()
    if region and region != "Todos":
        d = d[d["region"] == region]
    if prov and prov != "Todos":
        d = d[d["provincia"] == prov]
    if area and area != "Todos":
        d = d[d["area"] == area]
    if educ and educ != "Todos":
        d = d[d["nivel_educativo"] == educ]
    if quintil and quintil != "Todos":
        d = d[d["pobreza_quintil"] == quintil]
    return d

def kpi_card(title, value, subtitle="", color=C["purple_mid"], icon=""):
    return dbc.Card([
        dbc.CardBody([
            html.Div(icon, style={"fontSize":"1.4rem","marginBottom":"3px"}),
            html.Div(value, style={"fontSize":"1.9rem","fontWeight":"800",
                                   "color":color,"lineHeight":"1"}),
            html.Div(title, style={"fontSize":"0.72rem","fontWeight":"700",
                                   "color":C["gray_dark"],"marginTop":"3px",
                                   "textTransform":"uppercase","letterSpacing":"0.04em"}),
            html.Div(subtitle, style={"fontSize":"0.68rem","color":C["gray_mid"],"marginTop":"2px"}),
        ], style={"padding":"14px 12px"})
    ], style={"borderRadius":"12px","border":"none",
              "boxShadow":"0 2px 10px rgba(0,0,0,0.07)",
              "background":C["white"],"height":"100%"})


COLORSCALES = {
    "riesgo":    [[0,"#e1f5ee"],[0.4,"#f2b15a"],[0.7,"#ef9f27"],[1,"#c0392b"]],
    "acceso":    [[0,"#faeeda"],[0.4,"#5dcaa5"],[1,"#0f6e56"]],
    "prioridad": [[0,"#eeedfe"],[0.35,"#534ab7"],[0.65,"#26215c"],[1,"#c0392b"]],
}
CBAR_TITLES = {"riesgo":"Riesgo","acceso":"Acceso","prioridad":"Prioridad"}

def make_map(dff, color_col, map_id="map-generic"):
    """
    Build choropleth with 3 label layers:
      trace 1 → dept labels  (visible at zoom <5.5)
      trace 2 → prov labels  (visible at zoom 5.5–7.5)
      trace 3 → dist labels  (visible at zoom >7.5)
    Initial zoom 4.5 → show dept labels only.
    """
    # Filter centroids to visible ubigeos
    visible_ubigeos = set(dff["ubigeo"])

    fig = px.choropleth_mapbox(
        dff, geojson=geojson_dict, locations="ubigeo",
        color=color_col,
        color_continuous_scale=COLORSCALES[color_col],
        range_color=[0, 1],
        mapbox_style="carto-positron",
        zoom=4.5, center={"lat": -9.5, "lon": -75.5},
        opacity=0.78,
        custom_data=["district","provincia","region",
                     "riesgo","acceso","prioridad","semaforo"],
    )
    fig.update_traces(
        hovertemplate=(
            "<b>%{customdata[0]}</b><br>"
            "Provincia: %{customdata[1]}<br>"
            "Región: %{customdata[2]}<br>"
            "Riesgo: %{customdata[3]:.2f}  "
            "Acceso: %{customdata[4]:.2f}  "
            "Prioridad: %{customdata[5]:.2f}<br>"
            "Estado: %{customdata[6]}<extra></extra>"
        )
    )

    # ── Label layer 1: DEPARTAMENTOS (zoom < 5.5) ──
    fig.add_trace(go.Scattermapbox(
        lat=dept_cent["lat"], lon=dept_cent["lon"],
        text=dept_cent["nombdep"].str.title(),
        mode="text",
        textfont=dict(size=11, color=C["purple_dark"],
                      family="Inter, sans-serif"),
        hoverinfo="skip", showlegend=False,
        name="dept_labels", visible=True,
    ))

    # ── Label layer 2: PROVINCIAS (zoom 5.5–7.5) ──
    fig.add_trace(go.Scattermapbox(
        lat=prov_cent["lat"], lon=prov_cent["lon"],
        text=prov_cent["nombprov"].str.title(),
        mode="text",
        textfont=dict(size=9, color=C["purple_mid"],
                      family="Inter, sans-serif"),
        hoverinfo="skip", showlegend=False,
        name="prov_labels", visible=False,
    ))

    # ── Label layer 3: DISTRITOS (zoom > 7.5) ──
    # Only for visible subset (perf)
    dc_visible = dist_cent[dist_cent["ubigeo"].isin(visible_ubigeos)]
    fig.add_trace(go.Scattermapbox(
        lat=dc_visible["lat"], lon=dc_visible["lon"],
        text=dc_visible["nombdist"].str.title(),
        mode="text",
        textfont=dict(size=8, color=C["gray_dark"],
                      family="Inter, sans-serif"),
        hoverinfo="skip", showlegend=False,
        name="dist_labels", visible=False,
    ))

    fig.update_layout(
        coloraxis_colorbar=dict(
            title=CBAR_TITLES[color_col],
            tickfont=dict(size=10),
            thickness=14, len=0.6,
        ),
        margin=dict(l=0, r=0, t=0, b=0),
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif"),
        clickmode="event+select",
        uirevision="map",  # preserve view on data update
    )
    return fig


# ── ZOOM LABEL CALLBACK (shared logic) ───────────────────────────────────────
def _update_labels_from_zoom(relayout, current_fig):
    """Return Patch() updating label visibility based on zoom level."""
    if not relayout or not current_fig:
        return dash.no_update

    zoom = None
    if "mapbox.zoom" in relayout:
        zoom = relayout["mapbox.zoom"]
    elif "mapbox._derived" in relayout:
        zoom = relayout.get("mapbox._derived", {}).get("coordinates", None)
        if zoom is None:
            return dash.no_update
    else:
        return dash.no_update

    patched = Patch()
    # traces: 0=choropleth, 1=dept, 2=prov, 3=dist
    if zoom < 5.5:
        patched["data"][1]["visible"] = True
        patched["data"][2]["visible"] = False
        patched["data"][3]["visible"] = False
    elif zoom < 7.5:
        patched["data"][1]["visible"] = False
        patched["data"][2]["visible"] = True
        patched["data"][3]["visible"] = False
    else:
        patched["data"][1]["visible"] = False
        patched["data"][2]["visible"] = False
        patched["data"][3]["visible"] = True
    return patched


# ── FILTER BAR ───────────────────────────────────────────────────────────────
def make_filter_bar():
    return dbc.Row([
        dbc.Col([
            html.Label("Región / Departamento", style={"fontSize":"0.7rem","fontWeight":"700",
                        "color":C["gray_mid"],"textTransform":"uppercase"}),
            dcc.Dropdown(id="fil-region",
                options=[{"label":"Todos","value":"Todos"}]+
                        [{"label":r,"value":r} for r in REGIONS],
                value="Todos", clearable=False, style={"fontSize":"0.82rem"})
        ], width=3),
        dbc.Col([
            html.Label("Provincia", style={"fontSize":"0.7rem","fontWeight":"700",
                        "color":C["gray_mid"],"textTransform":"uppercase"}),
            dcc.Dropdown(id="fil-prov",
                options=[{"label":"Todos","value":"Todos"}]+
                        [{"label":p,"value":p} for p in PROVINCIAS],
                value="Todos", clearable=False, style={"fontSize":"0.82rem"})
        ], width=3),
        dbc.Col([
            html.Label("Área", style={"fontSize":"0.7rem","fontWeight":"700",
                        "color":C["gray_mid"],"textTransform":"uppercase"}),
            dcc.Dropdown(id="fil-area",
                options=[{"label":"Todos","value":"Todos"}]+
                        [{"label":a,"value":a} for a in AREAS],
                value="Todos", clearable=False, style={"fontSize":"0.82rem"})
        ], width=2),
        dbc.Col([
            html.Label("Nivel educativo", style={"fontSize":"0.7rem","fontWeight":"700",
                        "color":C["gray_mid"],"textTransform":"uppercase"}),
            dcc.Dropdown(id="fil-educ",
                options=[{"label":"Todos","value":"Todos"}]+
                        [{"label":e,"value":e} for e in EDUC],
                value="Todos", clearable=False, style={"fontSize":"0.82rem"})
        ], width=2),
        dbc.Col([
            html.Label("Mapa principal", style={"fontSize":"0.7rem","fontWeight":"700",
                        "color":C["gray_mid"],"textTransform":"uppercase"}),
            dcc.Dropdown(id="fil-mapa",
                options=[
                    {"label":"Brecha / Prioridad","value":"prioridad"},
                    {"label":"Riesgo","value":"riesgo"},
                    {"label":"Acceso","value":"acceso"},
                ],
                value="prioridad", clearable=False, style={"fontSize":"0.82rem"})
        ], width=2),
    ], className="g-2 align-items-end", style={
        "background":C["white"],"padding":"12px 20px",
        "borderBottom":f"2px solid {C['purple_pale']}",
        "boxShadow":"0 2px 8px rgba(0,0,0,0.05)"
    })

# ── TAB STYLES ────────────────────────────────────────────────────────────────
TS  = {"padding":"10px 20px","fontWeight":"600","borderRadius":"8px 8px 0 0",
       "fontSize":"0.85rem","color":C["gray_mid"],"border":"none","background":"transparent"}
TSS = {**TS, "color":C["purple_mid"],
       "borderBottom":f"3px solid {C['purple_mid']}","background":C["white"]}

# ── APP ───────────────────────────────────────────────────────────────────────
app = dash.Dash(__name__,
    external_stylesheets=[
        dbc.themes.BOOTSTRAP,
        "https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&display=swap"
    ],
    suppress_callback_exceptions=True,
    title="Violencia Psicológica · Perú")

app.layout = html.Div(
    style={"fontFamily":"Inter, sans-serif","background":C["bg"],"minHeight":"100vh"},
    children=[
        # HEADER
        html.Div([
            dbc.Container(fluid=True, children=[
                dbc.Row([
                    dbc.Col([
                        html.Span("🛡️ ", style={"fontSize":"1.6rem"}),
                        html.Span("La herida que no se ve",
                            style={"fontSize":"1.3rem","fontWeight":"800","color":C["white"]}),
                        html.Span(" · Violencia psicológica contra la mujer en el Perú",
                            style={"fontSize":"0.85rem","color":"rgba(255,255,255,0.72)","marginLeft":"6px"}),
                    ], width="auto"),
                    dbc.Col([
                        html.Div([
                            html.Span("30%", style={"fontWeight":"800","color":C["orange_light"]}),
                            html.Span(" afectadas alguna vez  ",
                                style={"fontSize":"0.78rem","color":"rgba(255,255,255,0.78)"}),
                            html.Span("70%", style={"fontWeight":"800","color":C["orange_light"]}),
                            html.Span(" no busca ayuda  ",
                                style={"fontSize":"0.78rem","color":"rgba(255,255,255,0.78)"}),
                            html.Span("1 de 3", style={"fontWeight":"800","color":C["green_light"]}),
                            html.Span(" casos no se denuncia",
                                style={"fontSize":"0.78rem","color":"rgba(255,255,255,0.78)"}),
                        ])
                    ], style={"textAlign":"right"}, className="d-none d-lg-block")
                ], align="center")
            ])
        ], style={
            "background":f"linear-gradient(135deg, {C['purple_dark']} 0%, {C['purple_mid']} 100%)",
            "padding":"13px 24px","boxShadow":"0 4px 16px rgba(0,0,0,0.2)"
        }),

        # FILTER BAR
        dbc.Container(fluid=True, children=[make_filter_bar()]),

        # TABS
        dbc.Container(fluid=True, style={"padding":"0 16px"}, children=[
            dcc.Tabs(id="main-tabs", value="tab-overview", children=[
                dcc.Tab(label="📊 Vista General",        value="tab-overview",   style=TS, selected_style=TSS),
                dcc.Tab(label="🧠 Análisis de Riesgo",  value="tab-riesgo",     style=TS, selected_style=TSS),
                dcc.Tab(label="🏥 Acceso a Servicios",  value="tab-servicios",  style=TS, selected_style=TSS),
                dcc.Tab(label="⚠️ Brechas y Priorización", value="tab-brechas", style=TS, selected_style=TSS),
            ], style={"borderBottom":"none","marginTop":"8px"}),
            html.Div(id="tab-content", style={"paddingBottom":"30px"}),
        ]),

        dcc.Store(id="store-click"),
    ]
)

# ── UPDATE PROVINCE DROPDOWN WHEN REGION CHANGES ─────────────────────────────
@app.callback(
    Output("fil-prov", "options"),
    Output("fil-prov", "value"),
    Input("fil-region", "value"),
)
def update_prov_options(region):
    if region == "Todos":
        provs = sorted(df_orig["provincia"].unique())
    else:
        provs = sorted(df_orig[df_orig["region"]==region]["provincia"].unique())
    opts = [{"label":"Todos","value":"Todos"}] + [{"label":p,"value":p} for p in provs]
    return opts, "Todos"

# ── RENDER TABS ───────────────────────────────────────────────────────────────
@app.callback(
    Output("tab-content","children"),
    Input("main-tabs","value"),
    Input("fil-region","value"),
    Input("fil-prov","value"),
    Input("fil-area","value"),
    Input("fil-educ","value"),
    Input("fil-mapa","value"),
)
def render_tab(tab, region, prov, area, educ, mapa_col):
    dff = filter_df(region, prov, area, educ, "Todos")
    if   tab == "tab-overview":  return build_overview(dff, mapa_col)
    elif tab == "tab-riesgo":    return build_riesgo(dff)
    elif tab == "tab-servicios": return build_servicios(dff)
    elif tab == "tab-brechas":   return build_brechas(dff)
    return html.Div()


# ═══════════════════════════════════════════════════════════════
# TAB 1 — VISTA GENERAL
# ═══════════════════════════════════════════════════════════════
def build_overview(dff, mapa_col):
    n_alto    = int((dff["riesgo"] > 0.55).sum())
    n_baja    = int((dff["acceso"] < 0.3).sum())
    n_crit    = int((dff["semaforo"] == "Crítico").sum())
    ries_nac  = f"{dff['riesgo'].mean()*100:.1f}%"

    # Scatter cuadrantes
    scatter = go.Figure()
    for sem, col in SEM_COLOR.items():
        sub = dff[dff["semaforo"]==sem]
        scatter.add_trace(go.Scatter(
            x=sub["acceso"], y=sub["riesgo"], mode="markers", name=sem,
            marker=dict(color=col, size=5, opacity=0.65),
            text=sub["district"],
            hovertemplate="<b>%{text}</b><br>Acceso:%{x:.2f} Riesgo:%{y:.2f}<extra></extra>"
        ))
    scatter.add_hline(y=0.55, line_dash="dot", line_color=C["gray_light"], line_width=1)
    scatter.add_vline(x=0.35, line_dash="dot", line_color=C["gray_light"], line_width=1)
    for txt, ax, ay in [
        ("🔴 Alto riesgo\nBajo acceso", 0.08, 0.88),
        ("🟢 Bajo riesgo\nAlto acceso", 0.75, 0.10),
        ("🟡 Alto riesgo\nAlto acceso", 0.75, 0.88),
        ("⚪ Bajo riesgo\nBajo acceso", 0.08, 0.10),
    ]:
        scatter.add_annotation(x=ax, y=ay, xref="paper", yref="paper",
            text=txt, showarrow=False, font=dict(size=9, color=C["gray_mid"]),
            align="center", bgcolor="rgba(255,255,255,0.6)", borderpad=3)
    scatter.update_layout(
        plot_bgcolor=C["white"], paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=40,r=10,t=30,b=40),
        xaxis_title="Acceso", yaxis_title="Riesgo",
        legend=dict(orientation="h", y=1.02, x=1, xanchor="right", font=dict(size=10)),
        font=dict(family="Inter, sans-serif", size=11),
        title=dict(text="Riesgo vs Acceso", font=dict(size=13, color=C["purple_dark"]), x=0)
    )

    # Histogram
    fig_hist = px.histogram(dff, x="riesgo", nbins=35,
        color_discrete_sequence=[C["purple_light"]],
        labels={"riesgo":"Score de riesgo","count":"Distritos"})
    fig_hist.update_layout(
        plot_bgcolor=C["white"], paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=30,r=10,t=30,b=30), bargap=0.04,
        font=dict(family="Inter, sans-serif", size=11),
        title=dict(text="Distribución del riesgo", font=dict(size=13, color=C["purple_dark"]), x=0)
    )

    fig_map = make_map(dff, mapa_col, "map-overview")

    return html.Div([
        dbc.Row([
            dbc.Col(kpi_card("Riesgo promedio",      ries_nac, "Nacional", C["red_risk"],    "📍"), md=3),
            dbc.Col(kpi_card("Distritos alto riesgo", f"{n_alto:,}", "Riesgo > 55%", C["orange_dark"], "⚠️"), md=3),
            dbc.Col(kpi_card("Baja cobertura",        f"{n_baja:,}", "Acceso < 30%", C["purple_mid"],  "🏥"), md=3),
            dbc.Col(kpi_card("Estado crítico",        f"{n_crit:,}", "Prioridad máx", C["sem_critico"],"🔴"), md=3),
        ], className="g-3 mb-3 mt-2"),
        dbc.Row([
            dbc.Col([
                html.Div([
                    html.Span("Mapa distrital — ", style={"fontWeight":"700","fontSize":"0.88rem","color":C["purple_dark"]}),
                    html.Span("zoom para ver departamentos → provincias → distritos",
                              style={"fontSize":"0.75rem","color":C["gray_mid"]}),
                ], style={"marginBottom":"6px"}),
                dcc.Graph(id="map-overview", figure=fig_map,
                          style={"height":"480px","borderRadius":"12px","overflow":"hidden"},
                          config={"displayModeBar":True, "modeBarButtonsToRemove":["select2d","lasso2d"]})
            ], md=7),
            dbc.Col([
                dcc.Graph(id="scatter-overview", figure=scatter,
                          style={"height":"230px","marginBottom":"8px"},
                          config={"displayModeBar":False}),
                dcc.Graph(id="hist-overview", figure=fig_hist,
                          style={"height":"230px"},
                          config={"displayModeBar":False}),
            ], md=5),
        ], className="g-3"),
        html.Div(id="district-panel-overview"),
    ])


# ── Zoom label callback — overview map ───────────────────────────────────────
@app.callback(
    Output("map-overview", "figure"),
    Input("map-overview", "relayoutData"),
    State("map-overview", "figure"),
    prevent_initial_call=True,
)
def zoom_labels_overview(relayout, fig):
    return _update_labels_from_zoom(relayout, fig)


# ── District click panel ──────────────────────────────────────────────────────
@app.callback(
    Output("district-panel-overview","children"),
    Input("map-overview","clickData"),
    State("fil-region","value"),
    State("fil-prov","value"),
    State("fil-area","value"),
    State("fil-educ","value"),
    prevent_initial_call=True,
)
def district_panel(clickData, region, prov, area, educ):
    if not clickData:
        return ""
    dff = filter_df(region, prov, area, educ, "Todos")
    try:
        ubigeo = str(clickData["points"][0]["location"]).zfill(6)
        row = dff[dff["ubigeo"]==ubigeo].iloc[0]
    except:
        return ""
    sc = SEM_COLOR.get(row["semaforo"], C["gray_mid"])
    return dbc.Card([dbc.CardBody([
        dbc.Row([
            dbc.Col([
                html.H5(row["district"], style={"fontWeight":"800","color":C["purple_dark"],"marginBottom":"2px"}),
                html.Div(f"Provincia: {row['provincia']} · Región: {row['region']} · {row['area']}",
                         style={"fontSize":"0.78rem","color":C["gray_mid"],"marginBottom":"12px"}),
                dbc.Row([
                    dbc.Col(kpi_card("Riesgo",    f"{row['riesgo']:.0%}", "", C["red_risk"]),    md=3),
                    dbc.Col(kpi_card("Acceso",    f"{row['acceso']:.0%}", "", C["green_dark"]),  md=3),
                    dbc.Col(kpi_card("Prioridad", f"{row['prioridad']:.0%}", "", C["purple_mid"]), md=3),
                    dbc.Col(kpi_card("Estado",    row["semaforo"], "", sc),                        md=3),
                ], className="g-2"),
            ], md=8),
            dbc.Col([
                html.Div("Perfil del distrito", style={"fontWeight":"700","fontSize":"0.85rem",
                                                       "color":C["purple_dark"],"marginBottom":"8px"}),
                html.Div(f"👥 Mujeres: {row['poblacion_mujeres']:,}", style={"fontSize":"0.8rem","marginBottom":"4px"}),
                html.Div(f"📚 Educación: {row['nivel_educativo']}", style={"fontSize":"0.8rem","marginBottom":"4px"}),
                html.Div(f"💰 Pobreza: {row['pobreza_pct']:.1f}% ({row['pobreza_quintil']})", style={"fontSize":"0.8rem","marginBottom":"4px"}),
                html.Div(f"🏢 CEM: {row['n_cem']}  🏠 Refugios: {row['n_refugios']}", style={"fontSize":"0.8rem","marginBottom":"4px"}),
                html.Div(f"📞 Línea 100: {'Sí' if row['linea100_cobertura'] else 'No'}", style={"fontSize":"0.8rem","marginBottom":"8px"}),
                dbc.Alert(f"💡 {row['recomendacion']}",
                    color="danger" if row["semaforo"]=="Crítico" else
                          "warning" if row["semaforo"]=="Medio" else "success",
                    style={"fontSize":"0.77rem","padding":"8px 12px","borderRadius":"8px","marginBottom":"0"})
            ], md=4),
        ])
    ])], style={"borderRadius":"12px","border":f"2px solid {sc}",
                "boxShadow":"0 4px 20px rgba(0,0,0,0.09)","marginTop":"16px"})


# ═══════════════════════════════════════════════════════════════
# TAB 2 — ANÁLISIS DE RIESGO
# ═══════════════════════════════════════════════════════════════
def build_riesgo(dff):
    fig_map = make_map(dff, "riesgo", "map-riesgo")

    def bar_chart(grouped, x_col, y_col, title, order=None):
        if order:
            grouped = grouped.reindex(order)
        grouped = grouped.reset_index()
        fig = px.bar(grouped, x=x_col, y=y_col,
            color=y_col, color_continuous_scale=[[0,"#5dcaa5"],[1,"#c0392b"]],
            text=grouped[y_col].round(2))
        fig.update_traces(textposition="outside", textfont_size=10)
        fig.update_layout(
            plot_bgcolor=C["white"], paper_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=30,r=10,t=35,b=40), showlegend=False,
            coloraxis_showscale=False,
            font=dict(family="Inter, sans-serif", size=11),
            title=dict(text=title, font=dict(size=13, color=C["purple_dark"]), x=0))
        return fig

    r_educ  = dff.groupby("nivel_educativo")["riesgo"].mean()
    r_quint = dff.groupby("pobreza_quintil")["riesgo"].mean()
    r_area  = dff.groupby("area")["riesgo"].mean().reset_index()

    fig_educ  = bar_chart(r_educ,  "nivel_educativo", "riesgo",
                          "Riesgo por nivel educativo", ["Muy bajo","Bajo","Medio","Alto"])
    fig_quint = bar_chart(r_quint, "pobreza_quintil", "riesgo",
                          "Riesgo por quintil de pobreza", ["Q1","Q2","Q3","Q4","Q5"])

    fig_area = px.bar(r_area, x="area", y="riesgo", color="area",
        color_discrete_map={"Urbana":C["purple_light"],"Rural":C["orange_mid"]},
        text=r_area["riesgo"].round(2))
    fig_area.update_traces(textposition="outside", textfont_size=11)
    fig_area.update_layout(
        plot_bgcolor=C["white"], paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=30,r=10,t=35,b=40), showlegend=False,
        font=dict(family="Inter, sans-serif", size=11),
        title=dict(text="Urbano vs Rural", font=dict(size=13, color=C["purple_dark"]), x=0))

    top_reg = dff.groupby("region")["riesgo"].mean().nlargest(10).index.tolist()
    fig_box = px.box(dff[dff["region"].isin(top_reg)], x="region", y="riesgo", color="region",
        color_discrete_sequence=px.colors.qualitative.Bold)
    fig_box.update_layout(
        plot_bgcolor=C["white"], paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=30,r=10,t=35,b=90), showlegend=False, xaxis_tickangle=-40,
        font=dict(family="Inter, sans-serif", size=10),
        title=dict(text="Top 10 regiones · distribución distrital",
                   font=dict(size=13, color=C["purple_dark"]), x=0))

    alto = dff[dff["riesgo"]>0.65]
    if len(alto):
        am  = alto["area"].mode().iloc[0]
        em  = alto["nivel_educativo"].mode().iloc[0]
        pob = alto["pobreza_pct"].mean()
        qm  = alto["pobreza_quintil"].mode().iloc[0]
    else:
        am, em, pob, qm = "Rural","Muy bajo", 60, "Q4"

    perfil = dbc.Card([
        dbc.CardHeader("🔴 Perfil de mayor riesgo",
            style={"background":C["red_risk"],"color":C["white"],"fontWeight":"700",
                   "fontSize":"0.88rem","border":"none","borderRadius":"10px 10px 0 0"}),
        dbc.CardBody([
            html.P("Mujeres con mayor probabilidad de violencia psicológica:",
                   style={"fontWeight":"700","color":C["gray_dark"],"fontSize":"0.83rem"}),
            dbc.ListGroup([
                dbc.ListGroupItem([html.B("📍 Área: "), am]),
                dbc.ListGroupItem([html.B("📚 Educación: "), em]),
                dbc.ListGroupItem([html.B("💰 Pobreza promedio: "), f"{pob:.1f}%"]),
                dbc.ListGroupItem([html.B("📊 Quintil frecuente: "), qm]),
            ], flush=True, style={"fontSize":"0.82rem","borderRadius":"8px"}),
        ])
    ], style={"borderRadius":"12px","border":"none","boxShadow":"0 2px 12px rgba(0,0,0,0.07)"})

    return html.Div([
        dbc.Row([
            dbc.Col([
                html.Div("Mapa de riesgo · zoom para etiquetas",
                    style={"fontWeight":"700","fontSize":"0.88rem","color":C["purple_dark"],"marginBottom":"6px"}),
                dcc.Graph(id="map-riesgo", figure=fig_map, style={"height":"480px"},
                          config={"displayModeBar":True,"modeBarButtonsToRemove":["select2d","lasso2d"]})
            ], md=7),
            dbc.Col([
                dbc.Row([
                    dbc.Col(dcc.Graph(figure=fig_educ, style={"height":"225px"},
                                     config={"displayModeBar":False}), md=6),
                    dbc.Col(dcc.Graph(figure=fig_area, style={"height":"225px"},
                                     config={"displayModeBar":False}), md=6),
                ], className="g-2 mb-2"),
                dcc.Graph(figure=fig_quint, style={"height":"225px"},
                          config={"displayModeBar":False}),
            ], md=5),
        ], className="g-3 mt-2"),
        dbc.Row([
            dbc.Col(dcc.Graph(figure=fig_box, style={"height":"300px"},
                              config={"displayModeBar":False}), md=8),
            dbc.Col(perfil, md=4),
        ], className="g-3 mt-1"),
    ])


@app.callback(
    Output("map-riesgo","figure"),
    Input("map-riesgo","relayoutData"),
    State("map-riesgo","figure"),
    prevent_initial_call=True,
)
def zoom_labels_riesgo(relayout, fig):
    return _update_labels_from_zoom(relayout, fig)


# ═══════════════════════════════════════════════════════════════
# TAB 3 — ACCESO A SERVICIOS
# ═══════════════════════════════════════════════════════════════
def build_servicios(dff):
    fig_map = make_map(dff, "acceso", "map-servicios")

    pct_cem   = (dff["n_cem"]>0).mean()*100
    pct_ref   = (dff["n_refugios"]>0).mean()*100
    pct_l100  = dff["linea100_cobertura"].mean()*100
    acc_avg   = dff["acceso"].mean()*100

    svc = pd.DataFrame({
        "Servicio": ["CEM","Refugios","Otros"],
        "Total":    [dff["n_cem"].sum(), dff["n_refugios"].sum(), dff["n_otros_servicios"].sum()],
    })
    fig_svc = px.bar(svc, x="Servicio", y="Total", color="Servicio",
        color_discrete_map={"CEM":C["purple_light"],"Refugios":C["green_mid"],"Otros":C["orange_mid"]},
        text="Total")
    fig_svc.update_traces(textposition="outside", textfont_size=11)
    fig_svc.update_layout(plot_bgcolor=C["white"], paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=30,r=10,t=35,b=30), showlegend=False,
        font=dict(family="Inter, sans-serif", size=11),
        title=dict(text="Total de servicios", font=dict(size=13,color=C["purple_dark"]),x=0))

    reg_cem = (dff.groupby("region")
               .apply(lambda x: (x["n_cem"]>0).mean()*100)
               .reset_index(name="pct_con_cem")
               .sort_values("pct_con_cem"))
    fig_reg = px.bar(reg_cem, x="pct_con_cem", y="region", orientation="h",
        color="pct_con_cem",
        color_continuous_scale=[[0,"#faeeda"],[0.5,"#5dcaa5"],[1,"#0f6e56"]],
        text=reg_cem["pct_con_cem"].round(0).astype(int).astype(str)+"%")
    fig_reg.update_traces(textposition="outside", textfont_size=9)
    fig_reg.update_layout(plot_bgcolor=C["white"], paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=10,r=40,t=35,b=30), coloraxis_showscale=False, height=420,
        font=dict(family="Inter, sans-serif", size=10),
        title=dict(text="% distritos con CEM por región",
                   font=dict(size=13,color=C["purple_dark"]),x=0))

    fig_tel = px.histogram(dff, x="acceso_telefonia_pct", color="area", nbins=25,
        color_discrete_map={"Urbana":C["purple_light"],"Rural":C["orange_mid"]},
        barmode="overlay", opacity=0.72,
        labels={"acceso_telefonia_pct":"% cobertura telefonía","count":"Distritos"})
    fig_tel.update_layout(plot_bgcolor=C["white"], paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=30,r=10,t=40,b=30),
        font=dict(family="Inter, sans-serif", size=11),
        title=dict(text="Cobertura telefonía por área",
                   font=dict(size=13,color=C["purple_dark"]),x=0))

    return html.Div([
        dbc.Row([
            dbc.Col(kpi_card("Acceso promedio",    f"{acc_avg:.1f}%",  "", C["green_dark"],   "📊"), md=3),
            dbc.Col(kpi_card("Distritos con CEM",  f"{pct_cem:.1f}%",  "", C["green_mid"],    "🏢"), md=3),
            dbc.Col(kpi_card("Con refugio",        f"{pct_ref:.1f}%",  "", C["orange_dark"],  "🏠"), md=3),
            dbc.Col(kpi_card("Cobertura Línea 100",f"{pct_l100:.1f}%","", C["purple_mid"],   "📞"), md=3),
        ], className="g-3 mb-3 mt-2"),
        dbc.Row([
            dbc.Col([
                html.Div("Mapa de acceso · zoom para etiquetas",
                    style={"fontWeight":"700","fontSize":"0.88rem","color":C["purple_dark"],"marginBottom":"6px"}),
                dcc.Graph(id="map-servicios", figure=fig_map, style={"height":"440px"},
                          config={"displayModeBar":True,"modeBarButtonsToRemove":["select2d","lasso2d"]})
            ], md=7),
            dbc.Col([
                dcc.Graph(figure=fig_svc, style={"height":"215px"},  config={"displayModeBar":False}),
                dcc.Graph(figure=fig_tel, style={"height":"215px"},  config={"displayModeBar":False}),
            ], md=5),
        ], className="g-3"),
        dbc.Row([
            dbc.Col(dcc.Graph(figure=fig_reg, config={"displayModeBar":False}), md=12),
        ], className="g-3 mt-1"),
    ])


@app.callback(
    Output("map-servicios","figure"),
    Input("map-servicios","relayoutData"),
    State("map-servicios","figure"),
    prevent_initial_call=True,
)
def zoom_labels_servicios(relayout, fig):
    return _update_labels_from_zoom(relayout, fig)


# ═══════════════════════════════════════════════════════════════
# TAB 4 — BRECHAS Y PRIORIZACIÓN
# ═══════════════════════════════════════════════════════════════
def build_brechas(dff):
    fig_map = make_map(dff, "prioridad", "map-brechas")

    # Semáforo donut
    sc = dff["semaforo"].value_counts().reset_index()
    sc.columns = ["semaforo","count"]
    fig_sem = px.pie(sc, names="semaforo", values="count",
        color="semaforo", color_discrete_map=SEM_COLOR, hole=0.55)
    fig_sem.update_traces(textinfo="percent+label", textfont_size=11)
    fig_sem.update_layout(paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=0,r=0,t=30,b=0), showlegend=False,
        font=dict(family="Inter, sans-serif",size=11),
        title=dict(text="Estado de distritos",
                   font=dict(size=13,color=C["purple_dark"]),x=0.5))

    # Brecha riesgo-acceso por región
    rg = (dff.groupby("region")
            .agg(riesgo=("riesgo","mean"), acceso=("acceso","mean"))
            .assign(brecha=lambda d: d["riesgo"]-d["acceso"])
            .sort_values("brecha")
            .reset_index())
    fig_gap = go.Figure()
    fig_gap.add_trace(go.Bar(x=rg["riesgo"], y=rg["region"], orientation="h",
        name="Riesgo", marker_color=C["red_risk"], opacity=0.85))
    fig_gap.add_trace(go.Bar(x=rg["acceso"], y=rg["region"], orientation="h",
        name="Acceso", marker_color=C["green_mid"], opacity=0.85))
    fig_gap.update_layout(barmode="group", plot_bgcolor=C["white"],
        paper_bgcolor="rgba(0,0,0,0)", margin=dict(l=10,r=30,t=35,b=30),
        legend=dict(orientation="h",y=1.08,x=0.5,xanchor="center"),
        xaxis_title="Score (0–1)", font=dict(family="Inter, sans-serif",size=10),
        title=dict(text="Riesgo vs Acceso por región",
                   font=dict(size=13,color=C["purple_dark"]),x=0), height=420)

    # Top 20 table
    cols_show = ["district","provincia","region","area",
                 "riesgo","acceso","prioridad","semaforo","n_cem","recomendacion"]
    top20 = (dff.nlargest(20,"prioridad")[cols_show]
               .assign(riesgo=lambda d: d["riesgo"].round(2),
                       acceso=lambda d: d["acceso"].round(2),
                       prioridad=lambda d: d["prioridad"].round(2)))
    table = dash_table.DataTable(
        data=top20.to_dict("records"),
        columns=[
            {"name":"Distrito","id":"district"}, {"name":"Provincia","id":"provincia"},
            {"name":"Región","id":"region"},     {"name":"Área","id":"area"},
            {"name":"Riesgo","id":"riesgo"},     {"name":"Acceso","id":"acceso"},
            {"name":"Prioridad","id":"prioridad"},{"name":"Estado","id":"semaforo"},
            {"name":"CEM","id":"n_cem"},          {"name":"Recomendación","id":"recomendacion"},
        ],
        style_table={"overflowX":"auto","borderRadius":"10px","overflow":"hidden"},
        style_header={"backgroundColor":C["purple_dark"],"color":"white",
                      "fontWeight":"700","fontSize":"0.75rem","padding":"10px 10px","border":"none"},
        style_cell={"fontSize":"0.75rem","padding":"7px 10px",
                    "fontFamily":"Inter, sans-serif","border":"none",
                    "whiteSpace":"normal","height":"auto","maxWidth":"200px"},
        style_data_conditional=[
            {"if":{"filter_query":'{semaforo} = "Crítico"'},
             "backgroundColor":"#fff0f0","color":C["red_risk"],"fontWeight":"700"},
            {"if":{"filter_query":'{semaforo} = "Medio"'},
             "backgroundColor":"#fffbf0","color":C["orange_dark"]},
            {"if":{"filter_query":'{semaforo} = "Adecuado"'},
             "backgroundColor":"#f0faf5","color":C["green_dark"]},
            {"if":{"row_index":"odd"}, "backgroundColor":"#fafafa"},
        ],
        page_size=10, sort_action="native", filter_action="native",
    )

    # Simulator
    simulator = dbc.Card([
        dbc.CardHeader("🔬 Simulador de política pública",
            style={"background":C["purple_mid"],"color":C["white"],"fontWeight":"700",
                   "fontSize":"0.88rem","border":"none","borderRadius":"10px 10px 0 0"}),
        dbc.CardBody([
            html.P("¿Qué ocurre si implemento nuevos servicios en distritos críticos?",
                   style={"fontSize":"0.82rem","color":C["gray_dark"],"marginBottom":"12px"}),
            dbc.Row([
                dbc.Col([
                    html.Label("Nuevos CEM a implementar",
                               style={"fontSize":"0.77rem","fontWeight":"600"}),
                    dcc.Slider(id="sim-cem", min=0, max=50, step=5, value=10,
                        marks={i:str(i) for i in range(0,51,10)},
                        tooltip={"placement":"bottom"}),
                ], md=6),
                dbc.Col([
                    html.Label("% mejora en cobertura de telefonía",
                               style={"fontSize":"0.77rem","fontWeight":"600"}),
                    dcc.Slider(id="sim-tel", min=0, max=30, step=5, value=10,
                        marks={i:str(i) for i in range(0,31,10)},
                        tooltip={"placement":"bottom"}),
                ], md=6),
            ], className="mb-3"),
            dbc.Button("▶ Simular impacto", id="sim-btn",
                style={"background":C["purple_mid"],"border":"none",
                       "borderRadius":"8px","fontSize":"0.82rem"}),
            html.Div(id="sim-output", style={"marginTop":"14px"}),
        ])
    ], style={"borderRadius":"12px","border":"none","boxShadow":"0 2px 12px rgba(0,0,0,0.07)"})

    return html.Div([
        dbc.Row([
            dbc.Col([
                html.Div("Mapa de priorización · zoom para etiquetas",
                    style={"fontWeight":"700","fontSize":"0.88rem","color":C["purple_dark"],"marginBottom":"6px"}),
                dcc.Graph(id="map-brechas", figure=fig_map, style={"height":"430px"},
                          config={"displayModeBar":True,"modeBarButtonsToRemove":["select2d","lasso2d"]})
            ], md=7),
            dbc.Col([
                dcc.Graph(figure=fig_sem, style={"height":"210px"}, config={"displayModeBar":False}),
                simulator,
            ], md=5),
        ], className="g-3 mt-2"),
        dbc.Row([
            dbc.Col(dcc.Graph(figure=fig_gap, config={"displayModeBar":False}), md=5),
            dbc.Col([
                html.Div([
                    html.Span("📋 Top 20 distritos críticos",
                        style={"fontWeight":"700","fontSize":"0.9rem","color":C["purple_dark"]}),
                    html.Span("  (filtrable · ordenable)",
                        style={"fontSize":"0.74rem","color":C["gray_mid"]}),
                ], style={"marginBottom":"8px"}),
                table,
            ], md=7),
        ], className="g-3 mt-1"),
    ])


@app.callback(
    Output("map-brechas","figure"),
    Input("map-brechas","relayoutData"),
    State("map-brechas","figure"),
    prevent_initial_call=True,
)
def zoom_labels_brechas(relayout, fig):
    return _update_labels_from_zoom(relayout, fig)


# ── SIMULATOR CALLBACK ────────────────────────────────────────────────────────
@app.callback(
    Output("sim-output","children"),
    Input("sim-btn","n_clicks"),
    State("sim-cem","value"),
    State("sim-tel","value"),
    State("fil-region","value"),
    State("fil-prov","value"),
    State("fil-area","value"),
    State("fil-educ","value"),
    prevent_initial_call=True,
)
def simulate(n, n_cem_new, pct_tel, region, prov, area, educ):
    dff = filter_df(region, prov, area, educ, "Todos")
    criticos = dff[dff["semaforo"]=="Crítico"]
    sin_cem  = criticos[criticos["n_cem"]==0].nlargest(n_cem_new, "prioridad")
    mejora   = min(n_cem_new*0.06, 0.25) + (pct_tel/100)*0.25*0.5
    antes    = criticos["prioridad"].mean() if len(criticos) else 0
    despues  = antes * (1 - mejora)
    reduccion = (antes - despues) / antes * 100 if antes else 0

    return dbc.Alert([
        html.H6("📊 Resultado de la simulación",
                style={"fontWeight":"800","marginBottom":"8px"}),
        dbc.Row([
            dbc.Col([
                html.Div(f"{len(sin_cem)}", style={"fontSize":"1.8rem","fontWeight":"800","color":C["green_dark"]}),
                html.Div("distritos críticos\ncon nuevo CEM", style={"fontSize":"0.75rem","color":C["gray_mid"]}),
            ], md=4),
            dbc.Col([
                html.Div(f"{reduccion:.1f}%", style={"fontSize":"1.8rem","fontWeight":"800","color":C["orange_dark"]}),
                html.Div("reducción en\nprioridad media", style={"fontSize":"0.75rem","color":C["gray_mid"]}),
            ], md=4),
            dbc.Col([
                html.Div(f"{despues:.2f}", style={"fontSize":"1.8rem","fontWeight":"800","color":C["purple_mid"]}),
                html.Div("nueva prioridad\nestimada", style={"fontSize":"0.75rem","color":C["gray_mid"]}),
            ], md=4),
        ])
    ], color="success",
       style={"borderRadius":"10px","border":"none","fontSize":"0.82rem"})


if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=8050)
