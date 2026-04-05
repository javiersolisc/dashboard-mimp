# app.py
import os
import numpy as np
import pandas as pd
import geopandas as gpd
import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, html, dcc, callback, Input, Output, dash_table

# 1. Cargar datos

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR  = os.path.join(PROJECT_DIR, "data")

# 1.1 GeoJSON de distritos
gdf_distritos = gpd.read_file(os.path.join(DATA_DIR, "distritos_peru.geojson"))
gdf_distritos = gdf_distritos.to_crs(epsg=4326)  # WGS84 para mapas web

# 1.2 Datos ficticios de riesgo (ya calculados por SAE)
np.random.seed(42)
gdf_distritos = gdf_distritos.merge(
    pd.DataFrame({
        "ubigeo": gdf_distritos["ubigeo"],
        "prob_violencia_psicologica": np.random.beta(0.6, 1.8, len(gdf_distritos)),
        "pobreza": np.random.beta(1, 2, len(gdf_distritos)),
        "urbano_rural": np.random.choice(["urbano", "rural"], len(gdf_distritos))
    }),
    on="ubigeo",
    how="left"
)

# 1.3 Servicios del MIMP
df_servicios = pd.read_csv(os.path.join(DATA_DIR, "servicios_mimp.csv"))

# 1.4 Calcular accesibilidad por distrito (simplificado)
# Vamos a simular "acceso" como suma de 1 / distancia^2 a cada servicio
# En la práctica usarías coordenadas reales y distancias

# Fijar centroides de distritos
gdf_distritos["centro"] = gdf_distritos.geometry.centroid

# Asumimos que todos los servicios están en el mismo punto (demo)
# En la vida real tendrías lat/lon de cada CEM, refugio, etc.
# Aquí simulamos 3 centros de servicio
serv_locs = [
    (gdf_distritos["centro"].x.mean(), gdf_distritos["centro"].y.mean()),
    (gdf_distritos["centro"].x.mean() + 0.3, gdf_distritos["centro"].y.mean()),
    (gdf_distritos["centro"].x.mean(), gdf_distritos["centro"].y.mean() + 0.3)
]

def compute_simple_access(row, beta=2):
    px, py = row["centro"].x, row["centro"].y
    acc = 0
    for sx, sy in serv_locs:
        d = ((px - sx)**2 + (py - sy)**2)**0.5
        d = max(d, 0.001)
        acc += 1 / (d**beta)
    return acc

gdf_distritos["acceso_servicio"] = gdf_distritos.apply(compute_simple_access, axis=1)

# Normalizar acceso entre 0 y 1
acceso_min, acceso_max = gdf_distritos["acceso_servicio"].min(), gdf_distritos["acceso_servicio"].max()
gdf_distritos["acceso_servicio_norm"] = (
    (gdf_distritos["acceso_servicio"] - acceso_min) / (acceso_max - acceso_min)
)

# 1.5 Construir indicador de brecha y prioridad
gdf_distritos["riesgo_norm"] = gdf_distritos["prob_violencia_psicologica"]
gdf_distritos["brecha"] = gdf_distritos["riesgo_norm"] - gdf_distritos["acceso_servicio_norm"]
gdf_distritos["prioridad"] = gdf_distritos["riesgo_norm"] * (1 - gdf_distritos["acceso_servicio_norm"])

# Clasificación semáforo
def clasificar(row):
    riesgo = row["riesgo_norm"]
    acceso = row["acceso_servicio_norm"]
    if riesgo > 0.6 and acceso < 0.4:
        return "🔴 Prioridad crítica"
    elif riesgo > 0.4 and acceso < 0.5:
        return "🟡 Prioridad media"
    elif riesgo < 0.3 and acceso > 0.5:
        return "🟢 Adecuado"
    else:
        return "🔵 Monitoreo"

gdf_distritos["categoria"] = gdf_distritos.apply(clasificar, axis=1)

# 1.6 Agrupar por región (ficticio)
# Por simplicidad, asumimos una columna "region" en el GeoJSON
if "region" not in gdf_distritos:
    n = len(gdf_distritos)
    regiones = ["Lima", "Callao", "Costa Norte", "Sierra", "Selva"]
    gdf_distritos["region"] = np.random.choice(regiones, n, p=[0.4,0.1,0.2,0.2,0.1])

# -----------------------------------------------------------------------
# 2. App Dash
# -----------------------------------------------------------------------

app = Dash(__name__, title="Violencia psicológica contra las mujeres - MIMP")

# 3. Layout: explicación + infografía + mapa + gráficos
app.layout = html.Div(
    style={"fontFamily": "Arial, sans-serif", "padding": "20px"},
    children=[
        # 3.1 Encabezado explicativo (infografía amigable)
        html.Div(
            style={"textAlign": "center", "marginBottom": "20px"},
            children=[
                html.H1("¿Dónde y cómo actuar frente a la violencia psicológica?"),
                html.P([
                    "Este dashboard muestra: ",
                    html.Span("riesgo", style={"fontWeight": "bold", "color": "red"}),
                    " de violencia psicológica contra las mujeres, ",
                    html.Span("acceso a servicios del MIMP", style={"fontWeight": "bold", "color": "blue"}),
                    " y ",
                    html.Span("brechas de prioridad", style={"fontWeight": "bold", "color": "purple"}),
                    " por distrito."
                ]),
                html.Hr(),
            ],
        ),

        # 3.2 Infografía amigable (conceptos clave)
        html.Div(
            style={
                "backgroundColor": "#f9f9f9",
                "padding": "15px",
                "borderRadius": "8px",
                "marginBottom": "20px",
                "display": "grid",
                "gridTemplateColumns": "repeat(auto-fit, minmax(300px, 1fr))",
                "gap": "15px"
            },
            children=[
                html.Div([
                    html.H4("¿Qué es la violencia psicológica?"),
                    html.P("Es el control emocional, el aislamiento, los celos excesivos, las humillaciones constantes, que afectan la dignidad y la salud mental de la mujer, aunque no deje huellas físicas visibles.")
                ]),
                html.Div([
                    html.H4("Riesgo"),
                    html.P("Indica qué tan probable es que una mujer en el distrito sufra violencia psicológica, según factores como edad, educación, pobreza y contexto.")
                ]),
                html.Div([
                    html.H4("Acceso a servicios"),
                    html.P("Mide cuán cerca y disponibles están los servicios del MIMP (CEM, refugios, línea 100) para las mujeres en cada distrito.")
                ]),
                html.Div([
                    html.H4("Brecha prioritaria"),
                    html.P("Muestra dónde hay mucho riesgo pero poca atención: son los lugares donde el Estado debería intervenir con más urgencia.")
                ]),
            ],
        ),

        # 3.3 Filtros globales
        html.Div(
            style={"display": "flex", "gap": "15px", "marginBottom": "20px"},
            children=[
                html.Div([
                    html.Label("Región:"),
                    dcc.Dropdown(
                        id="filtro-region",
                        options=[{"label": r, "value": r} for r in gdf_distritos["region"].unique()],
                        value=None,
                        placeholder="Todas las regiones",
                    ),
                ]),
                html.Div([
                    html.Label("Tipo de mapa:"),
                    dcc.RadioItems(
                        id="tipo-map",
                        options=[
                            {"label": "Riesgo", "value": "riesgo"},
                            {"label": "Acceso a servicios", "value": "acceso"},
                            {"label": "Brecha", "value": "brecha"},
                            {"label": "Prioridad de intervención", "value": "prioridad"},
                        ],
                        value="prioridad",
                        inline=True,
                    ),
                ]),
            ],
        ),

        # 3.4 Mapa principal
        html.Div(
            style={"height": "60vh", "marginBottom": "20px"},
            children=[
                dcc.Graph(
                    id="mapa-peru",
                    config={"displayModeBar": False},
                ),
            ],
        ),

        # 3.5 KPI principales
        html.Div(
            id="container-kpis",
            style={"display": "flex", "gap": "20px", "marginBottom": "20px"},
        ),

        # 3.6 Gráfico de riesgo vs acceso
        html.Div(
            style={"height": "40vh", "marginBottom": "20px"},
            children=[
                dcc.Graph(id="scatter-riesgo-acceso"),
            ],
        ),

        # 3.7 Tabla de distritos prioritarios
        html.Div(
            style={"marginBottom": "20px"},
            children=[
                html.H4("Top distritos con mayor prioridad"),
                dash_table.DataTable(
                    id="tabla-prioridad",
                    columns=[
                        {"name": "Distrito", "id": "distrito"},
                        {"name": "Riesgo", "id": "riesgo", "type": "numeric", "format": {"precision": 2}},
                        {"name": "Acceso", "id": "acceso", "type": "numeric", "format": {"precision": 2}},
                        {"name": "Prioridad", "id": "prioridad", "type": "numeric", "format": {"precision": 2}},
                        {"name": "Clasificación", "id": "categoria"},
                    ],
                    style_table={"height": "300px", "overflowY": "auto"},
                    style_cell={"textAlign": "left"},
                    sort_by=[{"column_id": "prioridad", "direction": "desc"}],
                ),
            ],
        ),
    ],
)

# 4. Callbacks

@callback(
    Output("mapa-peru", "figure"),
    Input("filtro-region", "value"),
    Input("tipo-map", "value"),
)
def update_map(region, tipo_mapa):
    # Filtrar por región
    if region:
        df_map = gdf_distritos[gdf_distritos["region"] == region]
    else:
        df_map = gdf_distritos.copy()

    # Columna a mapear
    if tipo_mapa == "riesgo":
        var = "riesgo_norm"
        title = "Riesgo de violencia psicológica"
        color_scale = "OrRd"
    elif tipo_mapa == "acceso":
        var = "acceso_servicio_norm"
        title = "Acceso a servicios del MIMP"
        color_scale = "Blues"
    elif tipo_mapa == "brecha":
        var = "brecha"
        title = "Brecha (Riesgo – Acceso)"
        color_scale = "RdBu"
    else:
        var = "prioridad"
        title = "Prioridad de intervención"
        color_scale = "PuRd"

    # Convertir geometría a geojson para choropleth
    df_map = df_map.drop(columns=["centro"])
    df_map = df_map.to_crs(epsg=4326)

    fig = px.choropleth(
        df_map,
        geojson=df_map.geometry.__geo_interface__,
        locations=df_map.index,
        color=var,
        color_continuous_scale=color_scale,
        labels={"risk": title},
    )
    fig.update_layout(
        title=title,
        margin={"l": 0, "r": 0, "t": 50, "b": 0},
        height=500,
        geo=dict(
            showframe=False,
            showcoastlines=False,
            projection_type="mercator",
        )
    )
    return fig

@callback(
    Output("container-kpis", "children"),
    Input("filtro-region", "value"),
)
def update_kpis(region):
    df = gdf_distritos.copy()
    if region:
        df = df[df["region"] == region]

    riesgo_prom = df["riesgo_norm"].mean()
    acc_prom = df["acceso_servicio_norm"].mean()
    n_criticos = df[df["categoria"] == "🔴 Prioridad crítica"].shape[0]

    kpis = [
        html.Div(
            style={"flex": "1", "textAlign": "center", "padding": "10px", "backgroundColor": "#fff", "borderRadius": "8px"},
            children=[
                html.H5("Riesgo promedio"),
                html.H2(f"{riesgo_prom:.2%}"),
            ]
        ),
        html.Div(
            style={"flex": "1", "textAlign": "center", "padding": "10px", "backgroundColor": "#fff", "borderRadius": "8px"},
            children=[
                html.H5("Acceso a servicios"),
                html.H2(f"{acc_prom:.2%}"),
            ]
        ),
        html.Div(
            style={"flex": "1", "textAlign": "center", "padding": "10px", "backgroundColor": "#fff", "borderRadius": "8px"},
            children=[
                html.H5("Distritos críticos"),
                html.H2(n_criticos),
            ]
        ),
    ]
    return kpis

@callback(
    Output("scatter-riesgo-acceso", "figure"),
    Input("filtro-region", "value"),
)
def update_scatter(region):
    df = gdf_distritos.copy()
    if region:
        df = df[df["region"] == region]

    df = df.fillna(0)
    fig = go.Figure()

    # Cuadrantes de riesgo vs acceso
    low_risk = df["riesgo_norm"] < 0.4
    high_risk = df["riesgo_norm"] >= 0.4
    low_acc  = df["acceso_servicio_norm"] < 0.5
    high_acc = df["acceso_servicio_norm"] >= 0.5

    colors = {
        "🔴 Alto riesgo / bajo acceso": "#d62728",
        "🟡 Bajo riesgo / bajo acceso": "#f5c71a",
        "🔵 Alto riesgo / alto acceso": "#1f77b4",
        "🟢 Bajo riesgo / alto acceso": "#2ca02c",
    }

    for label, cond in [
        ("🔴 Alto riesgo / bajo acceso",  high_risk & low_acc),
        ("🟡 Bajo riesgo / bajo acceso",  ~high_risk & low_acc),
        ("🔵 Alto riesgo / alto acceso",  high_risk & high_acc),
        ("🟢 Bajo riesgo / alto acceso",  ~high_risk & high_acc),
    ]:
        df_sub = df.loc[cond]
        if df_sub.empty:
            continue
        fig.add_trace(
            go.Scatter(
                x=df_sub["acceso_servicio_norm"],
                y=df_sub["riesgo_norm"],
                mode="markers",
                marker=dict(color=colors[label], size=8),
                name=label,
                text=df_sub["distrito"]))
               