import json
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import dash
from dash import dcc, html, Input, Output, State, dash_table, callback_context
import dash_bootstrap_components as dbc

# ── PALETTE (from the infographic) ─────────────────────────────────────────
COLORS = {
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
    "semaforo_critico": "#c0392b",
    "semaforo_medio":   "#ef9f27",
    "semaforo_adecuado":"#1d9e75",
}

SEMAFORO_COLOR = {
    "Crítico":  COLORS["semaforo_critico"],
    "Medio":    COLORS["semaforo_medio"],
    "Adecuado": COLORS["semaforo_adecuado"],
}

# ── LOAD DATA ───────────────────────────────────────────────────────────────
df_orig = pd.read_csv("data/datos_distritos.csv")
with open("data/peru_distritos.geojson") as f:
    geojson = json.load(f)

# Map id → geojson feature id (already numeric)
df_orig["id"] = df_orig["id"].astype(str)
for feat in geojson["features"]:
    feat["id"] = str(feat["properties"]["id"])

REGIONS  = sorted(df_orig["region"].unique())
AREAS    = ["Urbana", "Rural"]
EDUC     = ["Alto", "Medio", "Bajo", "Muy bajo"]
QUINTILES= ["Q1", "Q2", "Q3", "Q4", "Q5"]

# ── HELPER ──────────────────────────────────────────────────────────────────
def filter_df(region, area, educ, quintil):
    d = df_orig.copy()
    if region and region != "Todos":
        d = d[d["region"] == region]
    if area and area != "Todos":
        d = d[d["area"] == area]
    if educ and educ != "Todos":
        d = d[d["nivel_educativo"] == educ]
    if quintil and quintil != "Todos":
        d = d[d["pobreza_quintil"] == quintil]
    return d

def kpi_card(title, value, subtitle="", color=COLORS["purple_mid"], icon=""):
    return dbc.Card([
        dbc.CardBody([
            html.Div(icon, style={"fontSize":"1.5rem","marginBottom":"4px"}),
            html.Div(value, style={
                "fontSize":"2rem","fontWeight":"800",
                "color": color, "lineHeight":"1"}),
            html.Div(title, style={
                "fontSize":"0.78rem","fontWeight":"700",
                "color": COLORS["gray_dark"],"marginTop":"2px","textTransform":"uppercase",
                "letterSpacing":"0.05em"}),
            html.Div(subtitle, style={
                "fontSize":"0.7rem","color":COLORS["gray_mid"],"marginTop":"2px"})
        ], style={"padding":"16px 12px"})
    ], style={
        "borderRadius":"12px","border":"none",
        "boxShadow":"0 2px 12px rgba(0,0,0,0.07)",
        "background":COLORS["white"],"height":"100%"
    })

def choropleth_fig(dff, color_col, colorscale, title_bar):
    fig = px.choropleth_mapbox(
        dff, geojson=geojson, locations="id",
        color=color_col,
        color_continuous_scale=colorscale,
        range_color=[0, 1],
        mapbox_style="carto-positron",
        zoom=4.2, center={"lat": -9.5, "lon": -75.5},
        opacity=0.75,
        hover_data={
            "district": True, "region": True,
            "riesgo": ":.2f", "acceso": ":.2f",
            "prioridad": ":.2f", "semaforo": True,
            "id": False, color_col: False
        },
        labels={"riesgo":"Riesgo","acceso":"Acceso","prioridad":"Prioridad",
                "semaforo":"Estado","district":"Distrito","region":"Región"}
    )
    fig.update_layout(
        coloraxis_colorbar=dict(title=title_bar, tickfont=dict(size=10)),
        margin=dict(l=0, r=0, t=0, b=0),
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif"),
        clickmode="event+select"
    )
    return fig

# ── GLOBAL FILTER BAR ────────────────────────────────────────────────────────
filter_bar = dbc.Row([
    dbc.Col([
        html.Label("Región", style={"fontSize":"0.72rem","fontWeight":"700",
                    "color":COLORS["gray_mid"],"textTransform":"uppercase"}),
        dcc.Dropdown(
            id="fil-region",
            options=[{"label":"Todos","value":"Todos"}]+
                    [{"label":r,"value":r} for r in REGIONS],
            value="Todos", clearable=False,
            style={"fontSize":"0.82rem"}
        )
    ], width=3),
    dbc.Col([
        html.Label("Área", style={"fontSize":"0.72rem","fontWeight":"700",
                    "color":COLORS["gray_mid"],"textTransform":"uppercase"}),
        dcc.Dropdown(
            id="fil-area",
            options=[{"label":"Todos","value":"Todos"}]+
                    [{"label":a,"value":a} for a in AREAS],
            value="Todos", clearable=False,
            style={"fontSize":"0.82rem"}
        )
    ], width=2),
    dbc.Col([
        html.Label("Nivel educativo", style={"fontSize":"0.72rem","fontWeight":"700",
                    "color":COLORS["gray_mid"],"textTransform":"uppercase"}),
        dcc.Dropdown(
            id="fil-educ",
            options=[{"label":"Todos","value":"Todos"}]+
                    [{"label":e,"value":e} for e in EDUC],
            value="Todos", clearable=False,
            style={"fontSize":"0.82rem"}
        )
    ], width=3),
    dbc.Col([
        html.Label("Quintil de pobreza", style={"fontSize":"0.72rem","fontWeight":"700",
                    "color":COLORS["gray_mid"],"textTransform":"uppercase"}),
        dcc.Dropdown(
            id="fil-quintil",
            options=[{"label":"Todos","value":"Todos"}]+
                    [{"label":q,"value":q} for q in QUINTILES],
            value="Todos", clearable=False,
            style={"fontSize":"0.82rem"}
        )
    ], width=2),
    dbc.Col([
        html.Label("Mapa muestra", style={"fontSize":"0.72rem","fontWeight":"700",
                    "color":COLORS["gray_mid"],"textTransform":"uppercase"}),
        dcc.Dropdown(
            id="fil-mapa",
            options=[
                {"label":"Brecha (prioridad)","value":"prioridad"},
                {"label":"Riesgo","value":"riesgo"},
                {"label":"Acceso","value":"acceso"},
            ],
            value="prioridad", clearable=False,
            style={"fontSize":"0.82rem"}
        )
    ], width=2),
], className="g-2 align-items-end", style={
    "background":COLORS["white"],
    "padding":"12px 20px 12px 20px",
    "borderBottom":f"2px solid {COLORS['purple_pale']}",
    "boxShadow":"0 2px 8px rgba(0,0,0,0.05)"
})

# ── TABS CONTENT ─────────────────────────────────────────────────────────────
tab_style = {
    "padding":"10px 20px","fontWeight":"600","borderRadius":"8px 8px 0 0",
    "fontSize":"0.85rem","color":COLORS["gray_mid"],
    "border":"none","background":"transparent"
}
tab_selected_style = {
    **tab_style,
    "color":COLORS["purple_mid"],
    "borderBottom":f"3px solid {COLORS['purple_mid']}",
    "background":COLORS["white"]
}

# ── LAYOUT ───────────────────────────────────────────────────────────────────
app = dash.Dash(
    __name__,
    external_stylesheets=[
        dbc.themes.BOOTSTRAP,
        "https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&display=swap"
    ],
    suppress_callback_exceptions=True,
    title="Violencia Psicológica · Perú"
)

app.layout = html.Div(style={"fontFamily":"Inter, sans-serif","background":COLORS["bg"],"minHeight":"100vh"}, children=[

    # ── HEADER ──────────────────────────────────────────────────────────────
    html.Div([
        dbc.Container(fluid=True, children=[
            dbc.Row([
                dbc.Col([
                    html.Div("🛡️", style={"fontSize":"1.8rem","display":"inline","marginRight":"10px"}),
                    html.Span("La herida que no se ve", style={
                        "fontSize":"1.35rem","fontWeight":"800","color":COLORS["white"]}),
                    html.Span(" · Violencia psicológica contra la mujer en el Perú",
                        style={"fontSize":"0.9rem","color":"rgba(255,255,255,0.75)","marginLeft":"6px"}),
                ], width="auto"),
                dbc.Col([
                    html.Div([
                        html.Span("30%", style={"fontWeight":"800","color":COLORS["orange_light"]}),
                        html.Span(" mujeres alguna vez afectadas  ", style={"fontSize":"0.8rem","color":"rgba(255,255,255,0.8)"}),
                        html.Span("5 de 7", style={"fontWeight":"800","color":COLORS["green_light"]}),
                        html.Span(" casos en último año  ", style={"fontSize":"0.8rem","color":"rgba(255,255,255,0.8)"}),
                        html.Span("70%", style={"fontWeight":"800","color":COLORS["orange_light"]}),
                        html.Span(" no busca ayuda", style={"fontSize":"0.8rem","color":"rgba(255,255,255,0.8)"}),
                    ])
                ], style={"textAlign":"right"}, className="d-none d-lg-block")
            ], align="center")
        ])
    ], style={
        "background":f"linear-gradient(135deg, {COLORS['purple_dark']} 0%, {COLORS['purple_mid']} 100%)",
        "padding":"14px 24px","boxShadow":"0 4px 16px rgba(0,0,0,0.2)"
    }),

    # ── FILTER BAR ──────────────────────────────────────────────────────────
    dbc.Container(fluid=True, children=[filter_bar]),

    # ── TABS ────────────────────────────────────────────────────────────────
    dbc.Container(fluid=True, style={"padding":"0 16px"}, children=[
        dcc.Tabs(id="main-tabs", value="tab-overview", children=[
            dcc.Tab(label="📊 Vista General", value="tab-overview",
                    style=tab_style, selected_style=tab_selected_style),
            dcc.Tab(label="🧠 Análisis de Riesgo", value="tab-riesgo",
                    style=tab_style, selected_style=tab_selected_style),
            dcc.Tab(label="🏥 Acceso a Servicios", value="tab-servicios",
                    style=tab_style, selected_style=tab_selected_style),
            dcc.Tab(label="⚠️ Brechas y Priorización", value="tab-brechas",
                    style=tab_style, selected_style=tab_selected_style),
        ], style={"borderBottom":"none","marginTop":"8px"}),

        html.Div(id="tab-content", style={"paddingBottom":"30px"})
    ]),

    # Store for clicked district
    dcc.Store(id="store-click"),
])

# ── RENDER TABS ──────────────────────────────────────────────────────────────
@app.callback(
    Output("tab-content","children"),
    Input("main-tabs","value"),
    Input("fil-region","value"),
    Input("fil-area","value"),
    Input("fil-educ","value"),
    Input("fil-quintil","value"),
    Input("fil-mapa","value"),
)
def render_tab(tab, region, area, educ, quintil, mapa_col):
    dff = filter_df(region, area, educ, quintil)

    if tab == "tab-overview":
        return build_overview(dff, mapa_col)
    elif tab == "tab-riesgo":
        return build_riesgo(dff)
    elif tab == "tab-servicios":
        return build_servicios(dff)
    elif tab == "tab-brechas":
        return build_brechas(dff)
    return html.Div("Seleccione una pestaña")

# ═══════════════════════════════════════════════════════════════════════════
# TAB 1 – VISTA GENERAL
# ═══════════════════════════════════════════════════════════════════════════
def build_overview(dff, mapa_col):
    n_alto = int((dff["riesgo"] > 0.55).sum())
    n_baja_cob = int((dff["acceso"] < 0.3).sum())
    n_critico = int((dff["semaforo"] == "Crítico").sum())
    riesgo_nac = f"{dff['riesgo'].mean()*100:.1f}%"

    # Scatter quadrant
    scatter = go.Figure()
    for sem, col in SEMAFORO_COLOR.items():
        sub = dff[dff["semaforo"]==sem]
        scatter.add_trace(go.Scatter(
            x=sub["acceso"], y=sub["riesgo"],
            mode="markers",
            name=sem,
            marker=dict(color=col, size=6, opacity=0.7),
            text=sub["district"],
            hovertemplate="<b>%{text}</b><br>Acceso: %{x:.2f}<br>Riesgo: %{y:.2f}<extra></extra>"
        ))
    scatter.add_hline(y=0.55, line_dash="dot", line_color=COLORS["gray_light"], line_width=1)
    scatter.add_vline(x=0.35, line_dash="dot", line_color=COLORS["gray_light"], line_width=1)
    scatter.update_layout(
        plot_bgcolor=COLORS["white"], paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=40,r=10,t=30,b=40),
        xaxis_title="Acceso a servicios", yaxis_title="Riesgo",
        legend=dict(orientation="h", yanchor="bottom", y=1, xanchor="right", x=1,
                    font=dict(size=10)),
        font=dict(family="Inter, sans-serif", size=11),
        title=dict(text="Riesgo vs Acceso por distrito", font=dict(size=13,
                   color=COLORS["purple_dark"]), x=0, pad=dict(l=4))
    )
    # Quadrant annotations
    for txt, ax, ay in [
        ("🔴 Alto riesgo\nBajo acceso", 0.08, 0.88),
        ("🟢 Bajo riesgo\nAlto acceso", 0.75, 0.10),
        ("🟡 Alto riesgo\nAlto acceso", 0.75, 0.88),
        ("⚪ Bajo riesgo\nBajo acceso", 0.08, 0.10),
    ]:
        scatter.add_annotation(x=ax, y=ay, xref="paper", yref="paper",
            text=txt, showarrow=False, font=dict(size=9.5, color=COLORS["gray_mid"]),
            align="center", bgcolor="rgba(255,255,255,0.6)", borderpad=4)

    # Choropleth
    colorscale_map = {
        "riesgo":    [[0,"#e1f5ee"],[0.5,"#ef9f27"],[1,"#c0392b"]],
        "acceso":    [[0,"#faeeda"],[0.5,"#5dcaa5"],[1,"#0f6e56"]],
        "prioridad": [[0,"#eeedfe"],[0.4,"#534ab7"],[1,"#26215c"]],
    }
    title_map = {"riesgo":"Riesgo","acceso":"Acceso","prioridad":"Prioridad"}
    fig_map = choropleth_fig(dff, mapa_col, colorscale_map[mapa_col], title_map[mapa_col])

    # Histogram riesgo distribution
    fig_hist = px.histogram(dff, x="riesgo", nbins=30,
        color_discrete_sequence=[COLORS["purple_light"]],
        labels={"riesgo":"Puntaje de riesgo","count":"Distritos"})
    fig_hist.update_layout(
        plot_bgcolor=COLORS["white"], paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=30,r=10,t=30,b=30), bargap=0.05,
        font=dict(family="Inter, sans-serif", size=11),
        title=dict(text="Distribución del riesgo", font=dict(size=13,
                   color=COLORS["purple_dark"]), x=0)
    )

    return html.Div([
        # KPIs
        dbc.Row([
            dbc.Col(kpi_card("Riesgo promedio nacional", riesgo_nac,
                "% estimado de mujeres en riesgo", COLORS["red_risk"], "📍"), md=3),
            dbc.Col(kpi_card("Distritos de alto riesgo", f"{n_alto:,}",
                "Riesgo > 55%", COLORS["orange_dark"], "⚠️"), md=3),
            dbc.Col(kpi_card("Distritos con baja cobertura", f"{n_baja_cob:,}",
                "Acceso < 30%", COLORS["purple_mid"], "🏥"), md=3),
            dbc.Col(kpi_card("Distritos en estado crítico", f"{n_critico:,}",
                "Prioridad máxima", COLORS["semaforo_critico"], "🔴"), md=3),
        ], className="g-3 mb-3 mt-2"),

        # Map + Scatter
        dbc.Row([
            dbc.Col([
                html.Div("Mapa distrital", style={
                    "fontWeight":"700","fontSize":"0.9rem",
                    "color":COLORS["purple_dark"],"marginBottom":"6px"}),
                dcc.Graph(id="map-overview", figure=fig_map,
                          style={"height":"460px","borderRadius":"12px","overflow":"hidden"},
                          config={"displayModeBar":False})
            ], md=7),
            dbc.Col([
                dcc.Graph(id="scatter-overview", figure=scatter,
                          style={"height":"225px","marginBottom":"8px"},
                          config={"displayModeBar":False}),
                dcc.Graph(id="hist-overview", figure=fig_hist,
                          style={"height":"225px"},
                          config={"displayModeBar":False}),
            ], md=5),
        ], className="g-3"),

        # District panel (populated on click)
        html.Div(id="district-panel-overview"),
    ])

# ═══════════════════════════════════════════════════════════════════════════
# TAB 2 – ANÁLISIS DE RIESGO
# ═══════════════════════════════════════════════════════════════════════════
def build_riesgo(dff):
    # Map
    fig_map = choropleth_fig(dff, "riesgo",
        [[0,"#e1f5ee"],[0.4,"#f2b15a"],[0.7,"#ef9f27"],[1,"#c0392b"]], "Riesgo")

    # Bar: riesgo por nivel educativo
    educ_order = ["Muy bajo","Bajo","Medio","Alto"]
    r_educ = dff.groupby("nivel_educativo")["riesgo"].mean().reindex(educ_order).reset_index()
    fig_educ = px.bar(r_educ, x="nivel_educativo", y="riesgo",
        color="riesgo", color_continuous_scale=[[0,"#5dcaa5"],[1,"#c0392b"]],
        labels={"nivel_educativo":"Nivel educativo","riesgo":"Riesgo medio"},
        text=r_educ["riesgo"].round(2))
    fig_educ.update_traces(textposition="outside", textfont_size=11)
    fig_educ.update_layout(
        plot_bgcolor=COLORS["white"], paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=30,r=10,t=35,b=40), showlegend=False,
        coloraxis_showscale=False,
        font=dict(family="Inter, sans-serif",size=11),
        title=dict(text="Riesgo por nivel educativo",
                   font=dict(size=13,color=COLORS["purple_dark"]),x=0)
    )

    # Bar: riesgo por quintil de pobreza
    q_order = ["Q1","Q2","Q3","Q4","Q5"]
    r_quint = dff.groupby("pobreza_quintil")["riesgo"].mean().reindex(q_order).reset_index()
    fig_quint = px.bar(r_quint, x="pobreza_quintil", y="riesgo",
        color="riesgo", color_continuous_scale=[[0,"#5dcaa5"],[1,"#c0392b"]],
        labels={"pobreza_quintil":"Quintil de pobreza","riesgo":"Riesgo medio"},
        text=r_quint["riesgo"].round(2))
    fig_quint.update_traces(textposition="outside", textfont_size=11)
    fig_quint.update_layout(
        plot_bgcolor=COLORS["white"], paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=30,r=10,t=35,b=40), showlegend=False,
        coloraxis_showscale=False,
        font=dict(family="Inter, sans-serif",size=11),
        title=dict(text="Riesgo por quintil de pobreza",
                   font=dict(size=13,color=COLORS["purple_dark"]),x=0)
    )

    # Bar: riesgo por área
    r_area = dff.groupby("area")["riesgo"].mean().reset_index()
    fig_area = px.bar(r_area, x="area", y="riesgo",
        color="area",
        color_discrete_map={"Urbana":COLORS["purple_light"],"Rural":COLORS["orange_mid"]},
        labels={"area":"Área","riesgo":"Riesgo medio"},
        text=r_area["riesgo"].round(2))
    fig_area.update_traces(textposition="outside", textfont_size=11)
    fig_area.update_layout(
        plot_bgcolor=COLORS["white"], paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=30,r=10,t=35,b=40), showlegend=False,
        font=dict(family="Inter, sans-serif",size=11),
        title=dict(text="Riesgo: Urbano vs Rural",
                   font=dict(size=13,color=COLORS["purple_dark"]),x=0)
    )

    # Box: riesgo por región (top 10 with highest risk)
    top_reg = dff.groupby("region")["riesgo"].mean().nlargest(10).index.tolist()
    dff_top = dff[dff["region"].isin(top_reg)]
    fig_box = px.box(dff_top, x="region", y="riesgo", color="region",
        color_discrete_sequence=px.colors.qualitative.Bold,
        labels={"region":"Región","riesgo":"Riesgo"})
    fig_box.update_layout(
        plot_bgcolor=COLORS["white"], paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=30,r=10,t=35,b=90), showlegend=False,
        xaxis_tickangle=-40,
        font=dict(family="Inter, sans-serif",size=10),
        title=dict(text="Top 10 regiones con mayor riesgo (distribución distrital)",
                   font=dict(size=13,color=COLORS["purple_dark"]),x=0)
    )

    # Perfil de alto riesgo
    alto = dff[dff["riesgo"] > 0.65]
    if len(alto) > 0:
        area_mode = alto["area"].mode().iloc[0]
        educ_mode = alto["nivel_educativo"].mode().iloc[0]
        pob_avg = alto["pobreza_pct"].mean()
        quint_mode = alto["pobreza_quintil"].mode().iloc[0]
    else:
        area_mode, educ_mode, pob_avg, quint_mode = "Rural","Muy bajo",60,"Q4"

    perfil = dbc.Card([
        dbc.CardHeader(html.Span("🔴 Perfil de mayor riesgo", style={
            "fontWeight":"700","color":COLORS["white"],"fontSize":"0.9rem"}),
            style={"background":COLORS["red_risk"],"border":"none","borderRadius":"10px 10px 0 0"}),
        dbc.CardBody([
            html.P("Mujeres con mayor probabilidad de violencia psicológica:",
                   style={"fontWeight":"700","color":COLORS["gray_dark"],"fontSize":"0.85rem"}),
            dbc.ListGroup([
                dbc.ListGroupItem([
                    html.Span("📍 Área: ", style={"fontWeight":"700"}), area_mode]),
                dbc.ListGroupItem([
                    html.Span("📚 Educación: ", style={"fontWeight":"700"}), educ_mode]),
                dbc.ListGroupItem([
                    html.Span("💰 Pobreza promedio: ", style={"fontWeight":"700"}),
                    f"{pob_avg:.1f}%"]),
                dbc.ListGroupItem([
                    html.Span("📊 Quintil más frecuente: ", style={"fontWeight":"700"}), quint_mode]),
            ], flush=True, style={"fontSize":"0.82rem","borderRadius":"8px"}),
        ])
    ], style={"borderRadius":"12px","border":"none","boxShadow":"0 2px 12px rgba(0,0,0,0.07)"})

    return html.Div([
        dbc.Row([
            dbc.Col([
                html.Div("Mapa de riesgo por distrito", style={
                    "fontWeight":"700","fontSize":"0.9rem",
                    "color":COLORS["purple_dark"],"marginBottom":"6px"}),
                dcc.Graph(figure=fig_map, style={"height":"480px"},
                          config={"displayModeBar":False})
            ], md=7),
            dbc.Col([
                dbc.Row([
                    dbc.Col(dcc.Graph(figure=fig_educ, style={"height":"225px"},
                                     config={"displayModeBar":False}), md=6),
                    dbc.Col(dcc.Graph(figure=fig_area, style={"height":"225px"},
                                     config={"displayModeBar":False}), md=6),
                ], className="g-2 mb-2"),
                dbc.Row([
                    dbc.Col(dcc.Graph(figure=fig_quint, style={"height":"225px"},
                                     config={"displayModeBar":False}), md=12),
                ], className="g-2"),
            ], md=5),
        ], className="g-3 mt-2"),
        dbc.Row([
            dbc.Col(dcc.Graph(figure=fig_box, style={"height":"300px"},
                              config={"displayModeBar":False}), md=8),
            dbc.Col(perfil, md=4),
        ], className="g-3 mt-1"),
    ])

# ═══════════════════════════════════════════════════════════════════════════
# TAB 3 – ACCESO A SERVICIOS
# ═══════════════════════════════════════════════════════════════════════════
def build_servicios(dff):
    fig_map = choropleth_fig(dff, "acceso",
        [[0,"#faeeda"],[0.4,"#5dcaa5"],[1,"#0f6e56"]], "Acceso")

    # KPIs servicios
    pct_con_cem = dff[dff["n_cem"]>0].shape[0] / len(dff) * 100
    pct_con_refugio = dff[dff["n_refugios"]>0].shape[0] / len(dff) * 100
    pct_linea100 = dff["linea100_cobertura"].mean() * 100
    acceso_promedio = dff["acceso"].mean() * 100

    # Tipo de servicio
    servicios_counts = pd.DataFrame({
        "Servicio": ["CEM", "Refugios", "Otros servicios"],
        "Cantidad": [dff["n_cem"].sum(), dff["n_refugios"].sum(), dff["n_otros_servicios"].sum()],
        "Distritos con al menos 1": [
            (dff["n_cem"]>0).sum(), (dff["n_refugios"]>0).sum(),
            (dff["n_otros_servicios"]>0).sum()
        ]
    })
    fig_serv = px.bar(servicios_counts, x="Servicio", y="Cantidad",
        color="Servicio",
        color_discrete_map={"CEM":COLORS["purple_light"],"Refugios":COLORS["green_mid"],
                            "Otros servicios":COLORS["orange_mid"]},
        text="Cantidad")
    fig_serv.update_traces(textposition="outside", textfont_size=11)
    fig_serv.update_layout(
        plot_bgcolor=COLORS["white"], paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=30,r=10,t=35,b=30), showlegend=False,
        font=dict(family="Inter, sans-serif",size=11),
        title=dict(text="Total de servicios disponibles",
                   font=dict(size=13,color=COLORS["purple_dark"]),x=0)
    )

    # % distritos con CEM por región
    reg_cem = dff.groupby("region").apply(
        lambda x: (x["n_cem"]>0).mean()*100).reset_index()
    reg_cem.columns = ["region","pct_con_cem"]
    reg_cem = reg_cem.sort_values("pct_con_cem", ascending=True)
    fig_reg_cem = px.bar(reg_cem, x="pct_con_cem", y="region", orientation="h",
        color="pct_con_cem",
        color_continuous_scale=[[0,"#faeeda"],[0.5,"#5dcaa5"],[1,"#0f6e56"]],
        labels={"pct_con_cem":"% con CEM","region":""},
        text=reg_cem["pct_con_cem"].round(0).astype(int).astype(str)+"%")
    fig_reg_cem.update_traces(textposition="outside", textfont_size=9)
    fig_reg_cem.update_layout(
        plot_bgcolor=COLORS["white"], paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=10,r=40,t=35,b=30), showlegend=False,
        coloraxis_showscale=False, height=400,
        font=dict(family="Inter, sans-serif",size=10),
        title=dict(text="% distritos con CEM por región",
                   font=dict(size=13,color=COLORS["purple_dark"]),x=0)
    )

    # Telefonía y acceso
    fig_tel = px.histogram(dff, x="acceso_telefonia_pct", color="area", nbins=25,
        color_discrete_map={"Urbana":COLORS["purple_light"],"Rural":COLORS["orange_mid"]},
        barmode="overlay", opacity=0.7,
        labels={"acceso_telefonia_pct":"% cobertura telefonía","count":"Distritos"},
        title="Cobertura de telefonía por área")
    fig_tel.update_layout(
        plot_bgcolor=COLORS["white"], paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=30,r=10,t=40,b=30),
        font=dict(family="Inter, sans-serif",size=11),
        title_font=dict(size=13,color=COLORS["purple_dark"])
    )

    return html.Div([
        dbc.Row([
            dbc.Col(kpi_card("Acceso promedio", f"{acceso_promedio:.1f}%",
                "Score de acceso a servicios", COLORS["green_dark"], "📊"), md=3),
            dbc.Col(kpi_card("Distritos con CEM", f"{pct_con_cem:.1f}%",
                "Al menos 1 CEM activo", COLORS["green_mid"], "🏢"), md=3),
            dbc.Col(kpi_card("Distritos con refugio", f"{pct_con_refugio:.1f}%",
                "Al menos 1 refugio", COLORS["orange_dark"], "🏠"), md=3),
            dbc.Col(kpi_card("Cobertura Línea 100", f"{pct_linea100:.1f}%",
                "Distritos con acceso", COLORS["purple_mid"], "📞"), md=3),
        ], className="g-3 mb-3 mt-2"),

        dbc.Row([
            dbc.Col([
                html.Div("Mapa de acceso a servicios", style={
                    "fontWeight":"700","fontSize":"0.9rem",
                    "color":COLORS["purple_dark"],"marginBottom":"6px"}),
                dcc.Graph(figure=fig_map, style={"height":"440px"},
                          config={"displayModeBar":False})
            ], md=7),
            dbc.Col([
                dcc.Graph(figure=fig_serv, style={"height":"215px"},
                          config={"displayModeBar":False}),
                dcc.Graph(figure=fig_tel, style={"height":"215px"},
                          config={"displayModeBar":False}),
            ], md=5),
        ], className="g-3"),

        dbc.Row([
            dbc.Col(dcc.Graph(figure=fig_reg_cem,
                              config={"displayModeBar":False}), md=12),
        ], className="g-3 mt-1"),
    ])

# ═══════════════════════════════════════════════════════════════════════════
# TAB 4 – BRECHAS Y PRIORIZACIÓN
# ═══════════════════════════════════════════════════════════════════════════
def build_brechas(dff):
    fig_map = choropleth_fig(dff, "prioridad",
        [[0,"#eeedfe"],[0.35,"#534ab7"],[0.65,"#26215c"],[1,"#c0392b"]], "Prioridad")

    # Semáforo donut
    sem_counts = dff["semaforo"].value_counts().reset_index()
    sem_counts.columns = ["semaforo","count"]
    fig_sem = px.pie(sem_counts, names="semaforo", values="count",
        color="semaforo",
        color_discrete_map=SEMAFORO_COLOR,
        hole=0.55)
    fig_sem.update_traces(textinfo="percent+label", textfont_size=11)
    fig_sem.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=0,r=0,t=30,b=0),
        showlegend=False,
        font=dict(family="Inter, sans-serif",size=11),
        title=dict(text="Estado de distritos",
                   font=dict(size=13,color=COLORS["purple_dark"]),x=0.5)
    )

    # Brecha por región (riesgo - acceso)
    reg_gap = dff.groupby("region").agg(
        riesgo_mean=("riesgo","mean"),
        acceso_mean=("acceso","mean")
    ).reset_index()
    reg_gap["brecha"] = reg_gap["riesgo_mean"] - reg_gap["acceso_mean"]
    reg_gap = reg_gap.sort_values("brecha", ascending=True)
    fig_gap = go.Figure()
    fig_gap.add_trace(go.Bar(
        x=reg_gap["riesgo_mean"], y=reg_gap["region"],
        orientation="h", name="Riesgo",
        marker_color=COLORS["red_risk"], opacity=0.85))
    fig_gap.add_trace(go.Bar(
        x=reg_gap["acceso_mean"], y=reg_gap["region"],
        orientation="h", name="Acceso",
        marker_color=COLORS["green_mid"], opacity=0.85))
    fig_gap.update_layout(
        barmode="group", plot_bgcolor=COLORS["white"],
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=10,r=30,t=35,b=30),
        legend=dict(orientation="h", y=1.08, x=0.5, xanchor="center"),
        xaxis_title="Score (0–1)", yaxis_title="",
        font=dict(family="Inter, sans-serif",size=10),
        title=dict(text="Riesgo vs Acceso por región",
                   font=dict(size=13,color=COLORS["purple_dark"]),x=0),
        height=400
    )

    # Top 20 ranking table
    top20 = dff.nlargest(20, "prioridad")[
        ["district","region","area","riesgo","acceso","prioridad","semaforo","recomendacion","n_cem"]
    ].copy()
    top20["riesgo"] = top20["riesgo"].round(2)
    top20["acceso"] = top20["acceso"].round(2)
    top20["prioridad"] = top20["prioridad"].round(2)

    table = dash_table.DataTable(
        data=top20.to_dict("records"),
        columns=[
            {"name":"Distrito","id":"district"},
            {"name":"Región","id":"region"},
            {"name":"Área","id":"area"},
            {"name":"Riesgo","id":"riesgo"},
            {"name":"Acceso","id":"acceso"},
            {"name":"Prioridad","id":"prioridad"},
            {"name":"Estado","id":"semaforo"},
            {"name":"N° CEM","id":"n_cem"},
            {"name":"Recomendación","id":"recomendacion"},
        ],
        style_table={"overflowX":"auto","borderRadius":"10px","overflow":"hidden"},
        style_header={
            "backgroundColor":COLORS["purple_dark"],"color":"white",
            "fontWeight":"700","fontSize":"0.78rem","padding":"10px 12px",
            "border":"none"
        },
        style_cell={
            "fontSize":"0.78rem","padding":"8px 12px",
            "fontFamily":"Inter, sans-serif","border":"none",
            "whiteSpace":"normal","height":"auto","maxWidth":"220px"
        },
        style_data_conditional=[
            {"if":{"filter_query":'{semaforo} = "Crítico"'},
             "backgroundColor":"#fff0f0","color":COLORS["red_risk"],"fontWeight":"700"},
            {"if":{"filter_query":'{semaforo} = "Medio"'},
             "backgroundColor":"#fffbf0","color":COLORS["orange_dark"]},
            {"if":{"filter_query":'{semaforo} = "Adecuado"'},
             "backgroundColor":"#f0faf5","color":COLORS["green_dark"]},
            {"if":{"row_index":"odd"}, "backgroundColor":"#fafafa"},
        ],
        page_size=10,
        sort_action="native",
        filter_action="native",
    )

    # Simulator
    simulator = dbc.Card([
        dbc.CardHeader(html.Span("🔬 Simulador de política pública", style={
            "fontWeight":"700","color":COLORS["white"],"fontSize":"0.88rem"}),
            style={"background":COLORS["purple_mid"],"border":"none","borderRadius":"10px 10px 0 0"}),
        dbc.CardBody([
            html.P("¿Qué pasa si agrego servicios a los distritos críticos?",
                   style={"fontSize":"0.82rem","color":COLORS["gray_dark"],"marginBottom":"12px"}),
            dbc.Row([
                dbc.Col([
                    html.Label("Nº de nuevos CEM a implementar",
                               style={"fontSize":"0.78rem","fontWeight":"600"}),
                    dcc.Slider(id="sim-cem", min=0, max=50, step=5, value=10,
                        marks={i:str(i) for i in range(0,51,10)},
                        tooltip={"placement":"bottom"})
                ], md=6),
                dbc.Col([
                    html.Label("% mejora en acceso a telefonía",
                               style={"fontSize":"0.78rem","fontWeight":"600"}),
                    dcc.Slider(id="sim-tel", min=0, max=30, step=5, value=10,
                        marks={i:str(i) for i in range(0,31,10)},
                        tooltip={"placement":"bottom"})
                ], md=6),
            ], className="mb-3"),
            dbc.Button("▶ Simular impacto", id="sim-btn", color="primary",
                       style={"background":COLORS["purple_mid"],"border":"none",
                              "borderRadius":"8px","fontSize":"0.82rem"}),
            html.Div(id="sim-output", style={"marginTop":"14px"})
        ])
    ], style={"borderRadius":"12px","border":"none","boxShadow":"0 2px 12px rgba(0,0,0,0.07)"})

    return html.Div([
        dbc.Row([
            dbc.Col([
                html.Div("Mapa de priorización", style={
                    "fontWeight":"700","fontSize":"0.9rem",
                    "color":COLORS["purple_dark"],"marginBottom":"6px"}),
                dcc.Graph(figure=fig_map, style={"height":"420px"},
                          config={"displayModeBar":False})
            ], md=7),
            dbc.Col([
                dcc.Graph(figure=fig_sem, style={"height":"200px"},
                          config={"displayModeBar":False}),
                simulator,
            ], md=5),
        ], className="g-3 mt-2"),

        dbc.Row([
            dbc.Col(dcc.Graph(figure=fig_gap, style={"height":"400px"},
                              config={"displayModeBar":False}), md=5),
            dbc.Col([
                html.Div([
                    html.Span("📋 Top 20 distritos críticos",
                              style={"fontWeight":"700","fontSize":"0.9rem",
                                     "color":COLORS["purple_dark"]}),
                    html.Span(" (filtrable y ordenable)",
                              style={"fontSize":"0.75rem","color":COLORS["gray_mid"]}),
                ], style={"marginBottom":"8px"}),
                table
            ], md=7),
        ], className="g-3 mt-1"),
    ])

# ── SIMULATOR CALLBACK ────────────────────────────────────────────────────
@app.callback(
    Output("sim-output","children"),
    Input("sim-btn","n_clicks"),
    State("sim-cem","value"),
    State("sim-tel","value"),
    State("fil-region","value"),
    State("fil-area","value"),
    State("fil-educ","value"),
    State("fil-quintil","value"),
    prevent_initial_call=True
)
def simulate(n, n_cem_new, pct_tel, region, area, educ, quintil):
    dff = filter_df(region, area, educ, quintil)
    criticos = dff[dff["semaforo"]=="Crítico"].copy()

    # Assign new CEMs to highest-priority districts without CEM
    sin_cem = criticos[criticos["n_cem"]==0].nlargest(n_cem_new, "prioridad")
    mejora_acceso = min(n_cem_new * 0.06, 0.25)  # rough model
    mejora_tel = pct_tel / 100 * 0.25

    nueva_prioridad_criticos_antes = criticos["prioridad"].mean()
    nueva_prioridad_criticos_despues = nueva_prioridad_criticos_antes * (1 - mejora_acceso - mejora_tel * 0.5)
    distritos_rescatados = int(len(sin_cem))
    reduccion_pct = (nueva_prioridad_criticos_antes - nueva_prioridad_criticos_despues) / nueva_prioridad_criticos_antes * 100

    return dbc.Alert([
        html.H6("📊 Resultado de la simulación", style={"fontWeight":"800","marginBottom":"8px"}),
        dbc.Row([
            dbc.Col([
                html.Div(f"{distritos_rescatados}", style={"fontSize":"1.8rem","fontWeight":"800","color":COLORS["green_dark"]}),
                html.Div("distritos críticos\ncon nuevo CEM", style={"fontSize":"0.75rem","color":COLORS["gray_mid"]})
            ], md=4),
            dbc.Col([
                html.Div(f"{reduccion_pct:.1f}%", style={"fontSize":"1.8rem","fontWeight":"800","color":COLORS["orange_dark"]}),
                html.Div("reducción en\nprioridad media", style={"fontSize":"0.75rem","color":COLORS["gray_mid"]})
            ], md=4),
            dbc.Col([
                html.Div(f"{nueva_prioridad_criticos_despues:.2f}", style={"fontSize":"1.8rem","fontWeight":"800","color":COLORS["purple_mid"]}),
                html.Div("nueva prioridad\nestimada", style={"fontSize":"0.75rem","color":COLORS["gray_mid"]})
            ], md=4),
        ])
    ], color="success", style={"borderRadius":"10px","border":"none","fontSize":"0.82rem"})

# ── DISTRICT CLICK PANEL ──────────────────────────────────────────────────
@app.callback(
    Output("district-panel-overview","children"),
    Input("map-overview","clickData"),
    State("fil-region","value"),
    State("fil-area","value"),
    State("fil-educ","value"),
    State("fil-quintil","value"),
    prevent_initial_call=True
)
def show_district_panel(clickData, region, area, educ, quintil):
    if not clickData:
        return ""
    dff = filter_df(region, area, educ, quintil)
    try:
        loc_id = str(clickData["points"][0]["location"])
        row = dff[dff["id"]==loc_id].iloc[0]
    except:
        return ""

    sem_color = SEMAFORO_COLOR.get(row["semaforo"], COLORS["gray_mid"])

    return dbc.Card([
        dbc.CardBody([
            dbc.Row([
                dbc.Col([
                    html.H5(row["district"], style={"fontWeight":"800","color":COLORS["purple_dark"],"marginBottom":"2px"}),
                    html.Div(f"Región: {row['region']} · {row['area']}",
                             style={"fontSize":"0.8rem","color":COLORS["gray_mid"],"marginBottom":"12px"}),
                    dbc.Row([
                        dbc.Col(kpi_card("Riesgo", f"{row['riesgo']:.0%}", "",
                                         COLORS["red_risk"]), md=3),
                        dbc.Col(kpi_card("Acceso", f"{row['acceso']:.0%}", "",
                                         COLORS["green_dark"]), md=3),
                        dbc.Col(kpi_card("Prioridad", f"{row['prioridad']:.0%}", "",
                                         COLORS["purple_mid"]), md=3),
                        dbc.Col(kpi_card("Estado", row["semaforo"], "", sem_color), md=3),
                    ], className="g-2"),
                ], md=8),
                dbc.Col([
                    html.Div("Perfil del distrito", style={"fontWeight":"700","fontSize":"0.85rem",
                             "color":COLORS["purple_dark"],"marginBottom":"8px"}),
                    html.Div(f"👥 Mujeres: {row['poblacion_mujeres']:,}", style={"fontSize":"0.8rem","marginBottom":"4px"}),
                    html.Div(f"📚 Nivel educativo: {row['nivel_educativo']}", style={"fontSize":"0.8rem","marginBottom":"4px"}),
                    html.Div(f"💰 Pobreza: {row['pobreza_pct']:.1f}% ({row['pobreza_quintil']})", style={"fontSize":"0.8rem","marginBottom":"4px"}),
                    html.Div(f"🏢 CEM: {row['n_cem']}  🏠 Refugios: {row['n_refugios']}", style={"fontSize":"0.8rem","marginBottom":"4px"}),
                    html.Div(f"📞 Línea 100: {'Sí' if row['linea100_cobertura'] else 'No'}", style={"fontSize":"0.8rem","marginBottom":"8px"}),
                    dbc.Alert(f"💡 {row['recomendacion']}",
                              color="warning" if row["semaforo"]=="Medio" else
                                    "danger" if row["semaforo"]=="Crítico" else "success",
                              style={"fontSize":"0.78rem","padding":"8px 12px","borderRadius":"8px"})
                ], md=4),
            ])
        ])
    ], style={
        "borderRadius":"12px","border":f"2px solid {sem_color}",
        "boxShadow":"0 4px 20px rgba(0,0,0,0.10)","marginTop":"16px",
        "animation":"fadeIn 0.3s ease"
    })

if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=8050)
