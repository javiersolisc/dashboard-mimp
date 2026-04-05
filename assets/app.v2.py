import pandas as pd
import numpy as np
from dash import Dash, dcc, html, Input, Output
import plotly.express as px

# ======================
# CARGA DE DATOS
# ======================
df = pd.read_csv("data/data_distritos.csv")


import geopandas as gpd
import json

# Cargar shapefile
gdf = gpd.read_file("Distrito_INEI_2017/Distrito_INEI_2017.shp")

# Asegúrate que UBIGEO sea string
gdf["ubigeo"] = gdf["ubigeo"].astype(str)

# Cargar tu data
df = pd.read_csv("data/data_distritos.csv")
df["ubigeo"] = df["ubigeo"].astype(str)

# Normalización
df["riesgo"] = (df["riesgo"] - df["riesgo"].min()) / (df["riesgo"].max() - df["riesgo"].min())
df["acceso"] = (df["acceso"] - df["acceso"].min()) / (df["acceso"].max() - df["acceso"].min())

# Indicador clave
df["prioridad"] = df["riesgo"] * (1 - df["acceso"])

# Clasificación
df["categoria"] = np.where(df["prioridad"] > 0.6, "Crítico",
                   np.where(df["prioridad"] > 0.3, "Medio", "Bajo"))

print(gdf["ubigeo"].head())
print(df["ubigeo"].head())

# Merge
gdf = gdf.merge(df, on="ubigeo", how="left")

geojson = json.loads(gdf.to_json())

# ======================
# APP
# ======================
app = Dash(__name__)

# ======================
# LAYOUT
# ======================
app.layout = html.Div([

    html.H1("Dashboard de Violencia Psicológica y Acceso a Servicios"),

    dcc.Dropdown(
        options=[{"label": r, "value": r} for r in df["region"].unique()],
        placeholder="Filtrar por región",
        id="filtro_region"
    ),

    dcc.Tabs([

        # ======================
        # OVERVIEW
        # ======================
        dcc.Tab(label="Overview", children=[

            html.Div(id="kpis"),

            dcc.Graph(id="mapa_general"),

            dcc.Graph(id="scatter")
        ]),

        # ======================
        # RIESGO
        # ======================
        dcc.Tab(label="Riesgo", children=[
            dcc.Graph(id="mapa_riesgo"),
            dcc.Graph(id="riesgo_region")
        ]),

        # ======================
        # SERVICIOS
        # ======================
        dcc.Tab(label="Servicios", children=[
            dcc.Graph(id="mapa_acceso"),
            dcc.Graph(id="servicios_region")
        ]),

        # ======================
        # BRECHAS
        # ======================
        dcc.Tab(label="Brechas", children=[
            dcc.Graph(id="mapa_prioridad"),
            dcc.Graph(id="ranking")
        ])
    ])
])


# ======================
# FILTRO
# ======================
def filtrar(region):
    if region:
        return df[df["region"] == region]
    return df


# ======================
# KPIs
# ======================
@app.callback(
    Output("kpis", "children"),
    Input("filtro_region", "value")
)
def update_kpis(region):

    dff = filtrar(region)

    return html.Div([
        html.Div(f"Total distritos: {len(dff)}", className="kpi-box"),
        html.Div(f"Alto riesgo: {(dff['riesgo']>0.7).sum()}", className="kpi-box"),
        html.Div(f"Baja cobertura: {(dff['acceso']<0.3).sum()}", className="kpi-box"),
        html.Div(f"Críticos: {(dff['prioridad']>0.5).sum()}", className="kpi-box")
    ])


# ======================
# MAPA GENERAL
# ======================
@app.callback(
    Output("mapa_general", "figure"),
    Input("filtro_region", "value")
)
def mapa_general(region):

    dff = filtrar(region)

    fig = px.scatter_mapbox(
        dff,
        lat="lat",
        lon="lon",
        color="prioridad",
        size="prioridad",
        hover_name="distrito",
        zoom=4,
        height=600
    )

    fig.update_layout(mapbox_style="carto-positron")
    return fig


# ======================
# SCATTER
# ======================
@app.callback(
    Output("scatter", "figure"),
    Input("filtro_region", "value")
)
def scatter(region):

    dff = filtrar(region)

    fig = px.scatter(
        dff,
        x="acceso",
        y="riesgo",
        color="categoria",
        hover_name="distrito"
    )

    return fig


# ======================
# MAPA RIESGO
# ======================
@app.callback(
    Output("mapa_riesgo", "figure"),
    Input("filtro_region", "value")
)
def mapa_riesgo(region):

    dff = gdf.copy()

    if region:
        dff = dff[dff["region"] == region]

    fig = px.choropleth_mapbox(
        dff,
        geojson=json.loads(dff.to_json()),
        locations="ubigeo",
        featureidkey="properties.ubigeo",
        color="riesgo",
        hover_name="distrito",
        mapbox_style="carto-positron",
        zoom=4
    )

    return fig


# ======================
# RIESGO POR REGION
# ======================
@app.callback(
    Output("riesgo_region", "figure"),
    Input("filtro_region", "value")
)
def riesgo_region(region):

    dff = filtrar(region)

    temp = dff.groupby("region")["riesgo"].mean().reset_index()

    return px.bar(temp, x="region", y="riesgo")


# ======================
# MAPA ACCESO
# ======================
@app.callback(
    Output("mapa_acceso", "figure"),
    Input("filtro_region", "value")
)

def mapa_acceso(region):

    dff = gdf.copy()

    if region:
        dff = dff[dff["region"] == region]

    fig = px.choropleth_mapbox(
        dff,
        geojson=json.loads(dff.to_json()),
        locations="ubigeo",
        featureidkey="properties.ubigeo",
        color="acceso",
        hover_name="distrito",
        mapbox_style="carto-positron",
        zoom=4
    )

    return fig

# ======================
# SERVICIOS
# ======================
@app.callback(
    Output("servicios_region", "figure"),
    Input("filtro_region", "value")
)
def servicios_region(region):

    dff = filtrar(region)

    temp = dff.groupby("region")["n_servicios"].sum().reset_index()

    return px.bar(temp, x="region", y="n_servicios")


# ======================
# MAPA PRIORIDAD
# ======================
@app.callback(
    Output("mapa_prioridad", "figure"),
    Input("filtro_region", "value")
)

def mapa_prioridad(region):

    dff = gdf.copy()

    if region:
        dff = dff[dff["region"] == region]

    fig = px.choropleth_mapbox(
        dff,
        geojson=json.loads(dff.to_json()),
        locations="ubigeo",
        featureidkey="properties.ubigeo",
        color="prioridad",
        hover_name="distrito",
        mapbox_style="carto-positron",
        center={"lat": -9.19, "lon": -75.015},
        zoom=4,
        opacity=0.6
    )

    return fig

# ======================
# RANKING
# ======================
@app.callback(
    Output("ranking", "figure"),
    Input("filtro_region", "value")
)
def ranking(region):

    dff = filtrar(region)

    top = dff.sort_values("prioridad", ascending=False).head(10)

    return px.bar(top, x="prioridad", y="distrito", orientation="h")


# ======================
# RUN
# ======================
if __name__ == "__main__":
    app.run(debug=True)