"""
Hotel Bookings — Dashboard de Analítica y Visualización
Alejandro López | Entrega Individual
"""

import warnings
warnings.filterwarnings("ignore")
import logging
logging.getLogger("statsmodels").setLevel(logging.ERROR)

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE, LocallyLinearEmbedding
from sklearn.metrics import pairwise_distances
from scipy.stats import pearsonr, spearmanr, kendalltau, chi2_contingency
from statsmodels.tsa.stattools import grangercausalitytests
from scipy.optimize import minimize

# ─── PAGE CONFIG ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Hotel Analytics · Alejandro López",
    page_icon="🏨",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── GLOBAL STYLES ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ---------- tokens ---------- */
:root {
  --bg:          #0D1117;
  --surface:     #161B22;
  --surface2:    #1C2230;
  --border:      #30363D;
  --accent:      #58A6FF;
  --accent2:     #F78166;
  --accent3:     #3FB950;
  --accent4:     #D2A8FF;
  --text:        #E6EDF3;
  --muted:       #8B949E;
  --radius:      12px;
}

/* ---------- global ---------- */
html, body, [data-testid="stAppViewContainer"],
[data-testid="stApp"] {
  background-color: var(--bg) !important;
  color: var(--text);
  font-family: 'Inter', sans-serif;
}

/* ---------- sidebar ---------- */
[data-testid="stSidebar"] {
  background: var(--surface) !important;
  border-right: 1px solid var(--border);
}
[data-testid="stSidebar"] * { color: var(--text) !important; }
[data-testid="stSidebar"] .stSelectbox label,
[data-testid="stSidebar"] .stSlider label,
[data-testid="stSidebar"] .stRadio label  { color: var(--muted) !important; font-size: 0.75rem; text-transform: uppercase; letter-spacing: .05em; }

/* ---------- metric cards ---------- */
[data-testid="stMetric"] {
  background: var(--surface) !important;
  border: 1px solid var(--border);
  border-radius: var(--radius);
  padding: 1rem 1.25rem !important;
}
[data-testid="stMetricLabel"]  { color: var(--muted) !important; font-size:.75rem; text-transform:uppercase; letter-spacing:.06em; }
[data-testid="stMetricValue"]  { color: var(--text)  !important; font-size:1.9rem; font-weight:700; }
[data-testid="stMetricDelta"]  { font-size:.8rem; }

/* ---------- tabs ---------- */
button[data-baseweb="tab"] {
  background: transparent !important;
  color: var(--muted) !important;
  border-bottom: 2px solid transparent !important;
  font-size: .85rem; font-weight:600;
  text-transform: uppercase; letter-spacing:.06em;
  transition: color .2s, border-color .2s;
}
button[data-baseweb="tab"][aria-selected="true"] {
  color: var(--accent) !important;
  border-bottom-color: var(--accent) !important;
}
[data-testid="stTabsContent"] { padding-top: 1rem; }

/* ---------- plotly charts background ---------- */
.js-plotly-plot .plotly { border-radius: var(--radius); }

/* ---------- section headers ---------- */
.section-header {
  font-size:1.05rem; font-weight:700; color:var(--muted);
  text-transform:uppercase; letter-spacing:.08em;
  border-left: 3px solid var(--accent);
  padding-left:.6rem; margin: 1.5rem 0 .75rem;
}
.badge {
  display:inline-block; font-size:.7rem; font-weight:700;
  padding:.2rem .6rem; border-radius:999px;
  background: rgba(88,166,255,.15); color: var(--accent);
  border: 1px solid rgba(88,166,255,.3);
  margin-bottom:.5rem;
}
</style>
""", unsafe_allow_html=True)

# ─── COLOR PALETTE (Plotly) ─────────────────────────────────────────────────────
C_OK     = "#58A6FF"
C_CANCEL = "#F78166"
C_G3     = "#3FB950"
C_G4     = "#D2A8FF"
C_G5     = "#FFA657"
PALETTE  = [C_OK, C_CANCEL, C_G3, C_G4, C_G5, "#79C0FF", "#FF7B72"]

PLOT_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color="#E6EDF3", family="Inter"),
    margin=dict(l=20, r=20, t=40, b=20),
    legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor="#30363D", borderwidth=1),
    xaxis=dict(gridcolor="#21262D", linecolor="#30363D", zerolinecolor="#30363D"),
    yaxis=dict(gridcolor="#21262D", linecolor="#30363D", zerolinecolor="#30363D"),
)

def apply_theme(fig, title=""):
    fig.update_layout(**PLOT_LAYOUT)
    if title:
        fig.update_layout(title=dict(text=title, font=dict(size=14, color="#E6EDF3"), x=0))
    return fig

# ─── DATA LOAD & FEATURE ENGINEERING ───────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv("HotelData.csv")
    df["canceled"]       = (df["booking_status"] == "Canceled").astype(int)
    df["total_nights"]   = df["no_of_weekend_nights"] + df["no_of_week_nights"]
    df["total_guests"]   = df["no_of_adults"] + df["no_of_children"]
    df["revenue_est"]    = df["avg_price_per_room"] * df["total_nights"]
    df["arrival_season"] = pd.cut(df["arrival_month"], bins=[0,3,6,9,12],
                                   labels=["Invierno","Primavera","Verano","Otoño"])
    df["lead_time_cat"]  = pd.cut(df["lead_time"], bins=[-1,7,30,90,500],
                                   labels=["< 1 sem","1 sem-1 mes","1-3 meses","> 3 meses"])
    le = LabelEncoder()
    df["meal_enc"]   = le.fit_transform(df["type_of_meal_plan"])
    df["season_enc"] = le.fit_transform(df["arrival_season"].astype(str))
    return df

df_full = load_data()

MONTHS = {1:"Ene",2:"Feb",3:"Mar",4:"Abr",5:"May",6:"Jun",
          7:"Jul",8:"Ago",9:"Sep",10:"Oct",11:"Nov",12:"Dic"}

# ─── SIDEBAR ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center;padding:1rem 0 .5rem'>
      <div style='font-size:2.2rem'>🏨</div>
      <div style='font-weight:800;font-size:1.1rem;color:#E6EDF3'>Hotel Analytics</div>
      <div style='font-size:.75rem;color:#8B949E;margin-top:.2rem'>Alejandro López</div>
    </div>
    <hr style='border-color:#30363D;margin:.5rem 0 1rem'/>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-header">Filtros globales</div>', unsafe_allow_html=True)

    years_avail = sorted(df_full["arrival_year"].unique())
    sel_year = st.selectbox("📅 Año de llegada", ["Todos"] + [str(y) for y in years_avail])

    months_avail = sorted(df_full["arrival_month"].unique())
    sel_months = st.multiselect("🗓 Meses", options=months_avail,
                                 default=months_avail,
                                 format_func=lambda m: MONTHS[m])

    meal_options = df_full["type_of_meal_plan"].unique().tolist()
    sel_meal = st.multiselect("🍽 Plan de comida", options=meal_options, default=meal_options)

    sel_status = st.pills("Estado reserva",
                           options=["Todos", "Not_Canceled", "Canceled"],
                           default="Todos")

    st.markdown('<hr style="border-color:#30363D;margin:1rem 0"/>', unsafe_allow_html=True)
    st.markdown('<div class="section-header">Proyecciones</div>', unsafe_allow_html=True)
    proj_n = st.slider("Muestra para proyecciones", 500, 3000, 1500, 250)

    st.markdown('<hr style="border-color:#30363D;margin:1rem 0"/>', unsafe_allow_html=True)
    st.markdown("""
    <div style='font-size:.7rem;color:#8B949E;text-align:center'>
      Analítica y Visualización de Datos<br>
      Dataset: 25,811 reservas · 16 variables
    </div>
    """, unsafe_allow_html=True)

# ─── APPLY FILTERS ──────────────────────────────────────────────────────────────
df = df_full.copy()
if sel_year != "Todos":
    df = df[df["arrival_year"] == int(sel_year)]
if sel_months:
    df = df[df["arrival_month"].isin(sel_months)]
if sel_meal:
    df = df[df["type_of_meal_plan"].isin(sel_meal)]
if sel_status != "Todos":
    df = df[df["booking_status"] == sel_status]

if df.empty:
    st.warning("⚠️ Sin datos con los filtros actuales. Ajusta los filtros del panel izquierdo.")
    st.stop()

# ─── HEADER ─────────────────────────────────────────────────────────────────────
st.markdown(f"""
<div style='padding:.5rem 0 1.5rem'>
  <span class="badge">DASHBOARD INTERACTIVO</span>
  <h1 style='margin:0;font-size:2rem;font-weight:800;
             background:linear-gradient(90deg,#58A6FF,#D2A8FF);
             -webkit-background-clip:text;-webkit-text-fill-color:transparent'>
    Hotel Bookings — Analítica Completa
  </h1>
  <p style='color:#8B949E;margin:.3rem 0 0;font-size:.9rem'>
    {len(df):,} reservas seleccionadas · {df["canceled"].mean()*100:.1f}% tasa de cancelación
  </p>
</div>
""", unsafe_allow_html=True)

# ─── KPI CARDS ──────────────────────────────────────────────────────────────────
k1, k2, k3, k4, k5 = st.columns(5)
total_ref = len(df_full)
k1.metric("Total reservas",     f"{len(df):,}",
          f"{(len(df)/total_ref-1)*100:+.1f}% vs total")
k2.metric("Cancelaciones",      f"{df['canceled'].sum():,}",
          f"{df['canceled'].mean()*100:.1f}% tasa")
k3.metric("Precio promedio",    f"${df['avg_price_per_room'].mean():.0f}",
          f"σ = ${df['avg_price_per_room'].std():.0f}")
k4.metric("Lead time promedio", f"{df['lead_time'].mean():.0f} días",
          f"mediana {df['lead_time'].median():.0f} d")
k5.metric("Ingreso estimado",   f"${df['revenue_est'].sum()/1e6:.2f}M",
          f"avg ${df['revenue_est'].mean():.0f}/res")

st.markdown("<div style='height:.75rem'/>", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════════════════════
# TABS
# ════════════════════════════════════════════════════════════════════════════════
tabs = st.tabs([
    "📊 Visión General",
    "📈 Series & Estacionalidad",
    "🔵 Proyecciones",
    "🔗 Correlaciones",
    "🧮 Chi² & Granger",
    "🔍 Explorador",
])

# ────────────────────────────────────────────────────────────────────────────────
# TAB 1 — VISIÓN GENERAL
# ────────────────────────────────────────────────────────────────────────────────
with tabs[0]:
    row1_l, row1_r = st.columns([1, 1])

    # ── Donut: estado de reservas
    with row1_l:
        st.markdown('<div class="section-header">Estado de reservas</div>', unsafe_allow_html=True)
        cnt = df["booking_status"].value_counts().reset_index()
        cnt.columns = ["status", "count"]
        fig_donut = go.Figure(go.Pie(
            labels=cnt["status"], values=cnt["count"],
            hole=.60,
            marker=dict(colors=[C_OK, C_CANCEL],
                        line=dict(color="#0D1117", width=3)),
            textinfo="percent+label",
            textfont=dict(size=13),
        ))
        fig_donut.update_layout(**PLOT_LAYOUT, title="Distribución de estado",
                                 showlegend=False,
                                 annotations=[dict(text=f"<b>{len(df):,}</b><br>reservas",
                                                   x=.5,y=.5,showarrow=False,
                                                   font=dict(size=16,color="#E6EDF3"))])
        st.plotly_chart(fig_donut, use_container_width=True)

    # ── Barras: cancelación por plan de comida
    with row1_r:
        st.markdown('<div class="section-header">Cancelación por plan de comida</div>', unsafe_allow_html=True)
        meal_g = df.groupby("type_of_meal_plan", observed=True).agg(
            tasa=("canceled", "mean"), total=("canceled", "count")
        ).reset_index().sort_values("tasa", ascending=True)
        meal_g["tasa_pct"] = meal_g["tasa"] * 100
        fig_meal = px.bar(meal_g, y="type_of_meal_plan", x="tasa_pct",
                          orientation="h", color="tasa_pct",
                          color_continuous_scale=[[0,C_G3],[.5,C_G5],[1,C_CANCEL]],
                          labels={"tasa_pct":"% cancelación","type_of_meal_plan":"Plan"},
                          text=meal_g["tasa_pct"].apply(lambda v: f"{v:.1f}%"))
        fig_meal.update_traces(textposition="outside", marker_line_width=0)
        fig_meal.update_coloraxes(showscale=False)
        apply_theme(fig_meal, "Tasa de cancelación (%)"); fig_meal.update_layout(yaxis_title="")
        st.plotly_chart(fig_meal, use_container_width=True)

    row2_l, row2_r = st.columns([1, 1])

    # ── Histograma: lead time superpuesto
    with row2_l:
        st.markdown('<div class="section-header">Lead time vs cancelación</div>', unsafe_allow_html=True)
        fig_lt = go.Figure()
        for status, color, name in [("Not_Canceled", C_OK, "No cancelada"),
                                     ("Canceled",     C_CANCEL, "Cancelada")]:
            s = df[df["booking_status"] == status]["lead_time"]
            fig_lt.add_trace(go.Histogram(
                x=s, name=f"{name} (μ={s.mean():.0f}d)",
                nbinsx=40, opacity=.7,
                marker_color=color, marker_line_width=0,
            ))
        fig_lt.update_layout(**PLOT_LAYOUT, barmode="overlay",
                              title="Distribución Lead Time",
                              xaxis_title="Días", yaxis_title="Frecuencia")
        st.plotly_chart(fig_lt, use_container_width=True)

    # ── Box: precio por estado
    with row2_r:
        st.markdown('<div class="section-header">Precio por habitación</div>', unsafe_allow_html=True)
        df_box = df[df["avg_price_per_room"] > 0].copy()
        fig_box = px.box(df_box, x="booking_status", y="avg_price_per_room",
                         color="booking_status",
                         color_discrete_map={"Not_Canceled": C_OK, "Canceled": C_CANCEL},
                         points="outliers",
                         labels={"booking_status":"Estado","avg_price_per_room":"USD/noche"})
        fig_box.update_traces(marker_size=3, opacity=.8)
        apply_theme(fig_box, "Precio promedio por habitación")
        fig_box.update_layout(showlegend=False)
        st.plotly_chart(fig_box, use_container_width=True)

    # ── Outlier report full width
    st.markdown('<div class="section-header">Evaluación de outliers (método IQR)</div>', unsafe_allow_html=True)
    num_cols = df.select_dtypes(include=["int64","float64"]).columns.difference(
        ["canceled","meal_enc","season_enc","repeated_guest","required_car_parking_space"])
    rows = []
    for col in num_cols:
        Q1, Q3 = df[col].quantile([.25,.75])
        IQR = Q3 - Q1
        n_out = ((df[col] < Q1-1.5*IQR) | (df[col] > Q3+1.5*IQR)).sum()
        rows.append({"Variable":col,"Q1":round(Q1,2),"Q3":round(Q3,2),
                     "IQR":round(IQR,2),"Outliers":n_out,
                     "% del total":round(n_out/len(df)*100,2)})
    df_out = pd.DataFrame(rows).sort_values("Outliers",ascending=False)
    fig_out = px.bar(df_out, x="Variable", y="% del total", color="% del total",
                     color_continuous_scale=[[0,C_G3],[.4,C_G5],[1,C_CANCEL]],
                     text=df_out["% del total"].apply(lambda v:f"{v:.1f}%"),
                     labels={"% del total":"% outliers"})
    fig_out.update_traces(textposition="outside", marker_line_width=0)
    fig_out.update_coloraxes(showscale=False)
    apply_theme(fig_out, "Porcentaje de outliers por variable")
    st.plotly_chart(fig_out, use_container_width=True)

# ────────────────────────────────────────────────────────────────────────────────
# TAB 2 — SERIES & ESTACIONALIDAD
# ────────────────────────────────────────────────────────────────────────────────
with tabs[1]:
    # ── Reservas apiladas por mes
    st.markdown('<div class="section-header">Volumen mensual de reservas</div>', unsafe_allow_html=True)
    monthly = df.groupby("arrival_month", observed=True).agg(
        total=("canceled","count"), canceladas=("canceled","sum")
    ).reset_index()
    monthly["no_canceladas"] = monthly["total"] - monthly["canceladas"]
    monthly["mes"] = monthly["arrival_month"].map(MONTHS)
    monthly["tasa_cancelacion"] = monthly["canceladas"] / monthly["total"] * 100

    fig_monthly = go.Figure()
    fig_monthly.add_trace(go.Bar(x=monthly["mes"], y=monthly["no_canceladas"],
                                  name="No cancelada", marker_color=C_OK,
                                  marker_line_width=0))
    fig_monthly.add_trace(go.Bar(x=monthly["mes"], y=monthly["canceladas"],
                                  name="Cancelada", marker_color=C_CANCEL,
                                  marker_line_width=0))
    fig_monthly.update_layout(**PLOT_LAYOUT, barmode="stack",
                               title="Reservas por mes de llegada",
                               yaxis_title="Reservas", xaxis_title="Mes")
    st.plotly_chart(fig_monthly, use_container_width=True)

    col_a, col_b = st.columns(2)

    # ── Tasa cancelación por mes (línea)
    with col_a:
        st.markdown('<div class="section-header">Tasa de cancelación mensual</div>', unsafe_allow_html=True)
        fig_tasa = go.Figure(go.Scatter(
            x=monthly["mes"], y=monthly["tasa_cancelacion"].round(1),
            mode="lines+markers+text",
            line=dict(color=C_CANCEL, width=2.5),
            marker=dict(size=9, color=C_CANCEL, line=dict(color="#0D1117",width=2)),
            text=monthly["tasa_cancelacion"].apply(lambda v:f"{v:.1f}%"),
            textposition="top center", textfont=dict(size=10),
        ))
        apply_theme(fig_tasa, "% Cancelación por mes")
        fig_tasa.update_layout(yaxis_title="%", xaxis_title="Mes",
                                yaxis_range=[0, monthly["tasa_cancelacion"].max()*1.3])
        st.plotly_chart(fig_tasa, use_container_width=True)

    # ── Ingresos estimados por temporada
    with col_b:
        st.markdown('<div class="section-header">Ingreso estimado por temporada</div>', unsafe_allow_html=True)
        sea_rev = df[df["canceled"]==0].groupby("arrival_season", observed=True)["revenue_est"].agg(
            ["mean","sum"]).reset_index()
        sea_rev.columns = ["Temporada","Promedio","Total"]
        fig_sea = px.bar(sea_rev, x="Temporada", y="Promedio",
                         color="Temporada",
                         color_discrete_sequence=PALETTE,
                         text=sea_rev["Promedio"].apply(lambda v:f"${v:,.0f}"))
        fig_sea.update_traces(textposition="outside", marker_line_width=0)
        apply_theme(fig_sea, "Ingreso promedio por reserva confirmada (USD)")
        fig_sea.update_layout(showlegend=False, yaxis_title="USD")
        st.plotly_chart(fig_sea, use_container_width=True)

    # ── Heatmap mes × plan de comida
    st.markdown('<div class="section-header">Heatmap: % Cancelación mes × plan de comida</div>', unsafe_allow_html=True)
    pivot_heat = df.groupby(["arrival_month","type_of_meal_plan"], observed=True)["canceled"].mean().unstack() * 100
    pivot_heat.index = [MONTHS[m] for m in pivot_heat.index]
    fig_heat = px.imshow(pivot_heat,
                         color_continuous_scale=[[0,C_G3],[.4,C_G5],[1,C_CANCEL]],
                         text_auto=".0f",
                         aspect="auto",
                         labels=dict(x="Plan de comida", y="Mes", color="% Cancel."))
    apply_theme(fig_heat, "Tasa de cancelación (%) — Mes × Plan de comida")
    fig_heat.update_coloraxes(colorbar_title="% Cancel.")
    st.plotly_chart(fig_heat, use_container_width=True)

    # ── Lead time category distribution
    st.markdown('<div class="section-header">Lead time — distribución por categoría y estado</div>', unsafe_allow_html=True)
    lt_g = df.groupby(["lead_time_cat","booking_status"], observed=True).size().reset_index(name="n")
    lt_total = lt_g.groupby("lead_time_cat", observed=True)["n"].transform("sum")
    lt_g["pct"] = lt_g["n"] / lt_total * 100
    fig_lt2 = px.bar(lt_g, x="lead_time_cat", y="pct", color="booking_status",
                     color_discrete_map={"Not_Canceled":C_OK,"Canceled":C_CANCEL},
                     barmode="stack", text=lt_g["pct"].apply(lambda v:f"{v:.0f}%"),
                     labels={"lead_time_cat":"Anticipación","pct":"% del grupo",
                              "booking_status":"Estado"})
    fig_lt2.update_traces(textposition="inside", textfont_size=11, marker_line_width=0)
    apply_theme(fig_lt2, "Estado de reserva según anticipación de la reserva")
    fig_lt2.update_layout(yaxis_title="%")
    st.plotly_chart(fig_lt2, use_container_width=True)

# ────────────────────────────────────────────────────────────────────────────────
# TAB 3 — PROYECCIONES
# ────────────────────────────────────────────────────────────────────────────────
with tabs[2]:
    FEATURES = ["lead_time","avg_price_per_room","no_of_special_requests",
                "no_of_week_nights","no_of_weekend_nights","no_of_adults",
                "no_of_children","no_of_previous_cancellations",
                "no_of_previous_bookings_not_canceled",
                "total_nights","total_guests","meal_enc","season_enc",
                "required_car_parking_space"]

    @st.cache_data
    def prepare_projections(n_sample: int):
        df_w = df_full.copy()
        X = df_w[FEATURES].copy()
        y = df_w["canceled"].values
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        np.random.seed(42)
        idx = np.random.choice(len(X_scaled), size=min(n_sample, len(X_scaled)), replace=False)
        X_s, y_s = X_scaled[idx], y[idx]

        # PCA
        pca_full = PCA(random_state=42).fit(X_scaled)
        pca2 = PCA(n_components=2, random_state=42).fit(X_scaled)
        X_pca = pca2.transform(X_s)

        # t-SNE
        tsne = TSNE(n_components=2, perplexity=40, init="pca",
                    learning_rate="auto", random_state=42, n_jobs=-1)
        X_tsne = tsne.fit_transform(X_s)

        # LLE
        lle = LocallyLinearEmbedding(n_components=2, n_neighbors=15,
                                      method="standard", random_state=42, n_jobs=-1)
        X_lle = lle.fit_transform(X_s)

        # Sammon
        D_high = pairwise_distances(X_s[:500], metric="euclidean")
        D_high[D_high == 0] = 1e-10
        c = D_high.sum()
        def sammon_stress(flat, Dh, n, c):
            coords = flat.reshape(n, 2)
            Dl = pairwise_distances(coords, metric="euclidean")
            Dl[Dl == 0] = 1e-10
            return (1.0/c) * np.sum((Dh - Dl)**2 / Dh)
        x0 = PCA(n_components=2, random_state=42).fit_transform(X_s[:500]).flatten()
        res = minimize(sammon_stress, x0, args=(D_high, 500, c),
                       method="L-BFGS-B", options={"maxiter":300,"ftol":1e-6})
        X_samm = res.x.reshape(500, 2)
        y_samm = y_s[:500]

        return (pca_full, pca2, X_pca,
                X_tsne, X_lle, X_samm,
                y_s, y_samm, FEATURES,
                pca2.explained_variance_ratio_,
                pca_full.explained_variance_ratio_)

    with st.spinner("Calculando proyecciones (PCA, t-SNE, LLE, Sammon)…"):
        (pca_full, pca2, X_pca,
         X_tsne, X_lle, X_samm,
         y_s, y_samm, feat_names,
         ev2, ev_full) = prepare_projections(proj_n)

    # ── Scree plot + PCA biplot
    col_pca1, col_pca2 = st.columns([1,1])

    with col_pca1:
        st.markdown('<div class="section-header">PCA — Varianza explicada (Scree Plot)</div>', unsafe_allow_html=True)
        cumvar = np.cumsum(ev_full) * 100
        fig_scree = go.Figure()
        fig_scree.add_trace(go.Bar(x=list(range(1,len(ev_full)+1)),
                                    y=ev_full*100, name="Por componente",
                                    marker_color=C_OK, marker_line_width=0, opacity=.85))
        fig_scree.add_trace(go.Scatter(x=list(range(1,len(ev_full)+1)), y=cumvar,
                                        name="Acumulada", mode="lines+markers",
                                        line=dict(color=C_CANCEL,width=2),
                                        marker=dict(size=6)))
        fig_scree.add_hline(y=80, line_dash="dash", line_color="#8B949E", annotation_text="80%",
                             annotation_font_color="#8B949E")
        apply_theme(fig_scree, "Varianza explicada por componente")
        fig_scree.update_layout(xaxis_title="Componente", yaxis_title="%",
                                 yaxis_range=[0,105])
        st.plotly_chart(fig_scree, use_container_width=True)

    with col_pca2:
        st.markdown('<div class="section-header">PCA — Loadings (PC1 vs PC2)</div>', unsafe_allow_html=True)
        loadings = pd.DataFrame(pca2.components_.T, index=feat_names, columns=["PC1","PC2"])
        loadings["abs_pc1"] = loadings["PC1"].abs()
        loadings = loadings.sort_values("abs_pc1", ascending=False)
        fig_load = go.Figure()
        fig_load.add_trace(go.Bar(x=loadings.index, y=loadings["PC1"],
                                   name="PC1", marker_color=C_OK, opacity=.85, marker_line_width=0))
        fig_load.add_trace(go.Bar(x=loadings.index, y=loadings["PC2"],
                                   name="PC2", marker_color=C_G4, opacity=.85, marker_line_width=0))
        apply_theme(fig_load, "Contribución de variables a PC1 y PC2")
        fig_load.update_layout(barmode="group", xaxis_title="", yaxis_title="Loading",
                                xaxis_tickangle=-40)
        st.plotly_chart(fig_load, use_container_width=True)

    # ── PCA scatter biplot
    st.markdown('<div class="section-header">PCA — Proyección 2D + vectores de variables</div>', unsafe_allow_html=True)
    df_pca = pd.DataFrame({"x":X_pca[:,0],"y":X_pca[:,1],
                            "clase":["Cancelada" if yy else "No cancelada" for yy in y_s]})
    fig_pca = px.scatter(df_pca, x="x", y="y", color="clase",
                          color_discrete_map={"No cancelada":C_OK,"Cancelada":C_CANCEL},
                          opacity=.45, size_max=6,
                          labels={"x":f"PC1 ({ev2[0]*100:.1f}% var)",
                                  "y":f"PC2 ({ev2[1]*100:.1f}% var)"})
    scale = 3.5
    for i, feat in enumerate(feat_names):
        fig_pca.add_annotation(
            x=pca2.components_[0,i]*scale, y=pca2.components_[1,i]*scale,
            ax=0, ay=0, arrowhead=3, arrowwidth=1.5,
            arrowcolor="#8B949E",
            text=feat, font=dict(size=9, color="#E6EDF3"),
            bgcolor="rgba(22,27,34,.7)",
        )
    apply_theme(fig_pca, "PCA Biplot — Observaciones y variables")
    fig_pca.update_traces(marker_size=4)
    st.plotly_chart(fig_pca, use_container_width=True)

    # ── t-SNE / LLE / Sammon
    st.markdown('<div class="section-header">Proyecciones no lineales — Comparación</div>', unsafe_allow_html=True)
    col_t, col_l, col_s = st.columns(3)

    def proj_scatter(X2d, yy, title, ev_text=""):
        df_p = pd.DataFrame({"x":X2d[:,0],"y":X2d[:,1],
                              "clase":["Cancelada" if v else "No cancelada" for v in yy]})
        fig = px.scatter(df_p, x="x", y="y", color="clase",
                          color_discrete_map={"No cancelada":C_OK,"Cancelada":C_CANCEL},
                          opacity=.45)
        fig.update_traces(marker_size=3)
        apply_theme(fig, title)
        fig.update_layout(xaxis_title=ev_text if ev_text else "Dim 1",
                           yaxis_title="Dim 2", showlegend=True,
                           height=360)
        return fig

    with col_t:
        st.plotly_chart(proj_scatter(X_tsne, y_s, "t-SNE"), use_container_width=True)
    with col_l:
        st.plotly_chart(proj_scatter(X_lle, y_s, "LLE"), use_container_width=True)
    with col_s:
        st.plotly_chart(proj_scatter(X_samm, y_samm, f"Mapa de Sammon (n=500)"),
                        use_container_width=True)

    st.info("💡 **PCA** preserva varianza global (lineal). **t-SNE** y **LLE** preservan estructura local (no lineal). **Sammon** minimiza el estrés de distancias relativas entre todos los pares de puntos.")

# ────────────────────────────────────────────────────────────────────────────────
# TAB 4 — CORRELACIONES
# ────────────────────────────────────────────────────────────────────────────────
with tabs[3]:
    NUM_VARS = ["lead_time","avg_price_per_room","no_of_special_requests",
                "no_of_week_nights","no_of_weekend_nights","no_of_adults",
                "no_of_previous_cancellations","no_of_previous_bookings_not_canceled",
                "total_nights","total_guests","canceled"]

    col_m, col_s2 = st.columns([1,1])

    with col_m:
        st.markdown('<div class="section-header">Método de correlación</div>', unsafe_allow_html=True)
        corr_method = st.radio("", ["pearson","spearman","kendall"],
                                format_func=lambda m: {"pearson":"Pearson r",
                                                        "spearman":"Spearman ρ",
                                                        "kendall":"Kendall τ"}[m],
                                horizontal=True)
        corr_mat = df[NUM_VARS].corr(method=corr_method)
        mask_upper = np.triu(np.ones_like(corr_mat, dtype=bool), k=1)
        corr_display = corr_mat.where(~mask_upper)

        fig_corr = px.imshow(corr_display, text_auto=".2f",
                              color_continuous_scale="RdBu_r",
                              zmin=-1, zmax=1, aspect="auto",
                              color_continuous_midpoint=0)
        apply_theme(fig_corr, f"Matriz de correlación — {corr_method.capitalize()}")
        fig_corr.update_coloraxes(colorbar_title="r/ρ/τ")
        fig_corr.update_traces(textfont_size=9)
        st.plotly_chart(fig_corr, use_container_width=True)

    with col_s2:
        st.markdown('<div class="section-header">Correlación con variable objetivo (canceled)</div>', unsafe_allow_html=True)
        results = []
        for col in NUM_VARS:
            if col == "canceled": continue
            rp, pp = pearsonr(df[col], df["canceled"])
            rs, ps = spearmanr(df[col], df["canceled"])
            rk, pk = kendalltau(df[col], df["canceled"])
            results.append({"Variable":col,
                             "Pearson r":round(rp,4), "p_p":round(pp,4),
                             "Spearman ρ":round(rs,4), "p_s":round(ps,4),
                             "Kendall τ":round(rk,4), "p_k":round(pk,4)})
        df_cr = pd.DataFrame(results).sort_values("Pearson r", key=abs, ascending=False)

        fig_cr = go.Figure()
        x = df_cr["Variable"]
        w = 0.27
        for i, (col, color, name) in enumerate([
            ("Pearson r", C_OK, "Pearson r"),
            ("Spearman ρ", C_G4, "Spearman ρ"),
            ("Kendall τ", C_G3, "Kendall τ"),
        ]):
            fig_cr.add_trace(go.Bar(name=name, x=x, y=df_cr[col],
                                     marker_color=color, opacity=.87,
                                     marker_line_width=0,
                                     offset=(-w + i*w)))
        fig_cr.add_hline(y=0, line_color="#30363D")
        apply_theme(fig_cr, "Coeficientes vs 'canceled'")
        fig_cr.update_layout(barmode="group", xaxis_tickangle=-40,
                              yaxis_title="Coeficiente", xaxis_title="")
        st.plotly_chart(fig_cr, use_container_width=True)

    # ── Tabla resumen
    st.markdown('<div class="section-header">Tabla de coeficientes y p-valores</div>', unsafe_allow_html=True)
    df_show = df_cr.copy()
    df_show["Significativa"] = df_show["p_p"].apply(lambda p: "✅ Sí" if p < 0.05 else "❌ No")
    st.dataframe(df_show[["Variable","Pearson r","p_p","Spearman ρ","p_s","Kendall τ","p_k","Significativa"]]
                 .rename(columns={"p_p":"p (P)","p_s":"p (S)","p_k":"p (K)"}),
                 use_container_width=True, hide_index=True)

    # ── Scatter interactivo
    st.markdown('<div class="section-header">Scatter interactivo — variable vs cancelación</div>', unsafe_allow_html=True)
    sel_var = st.selectbox("Variable X:", [v for v in NUM_VARS if v != "canceled"])
    fig_sc = px.scatter(df.sample(min(2000, len(df)), random_state=1),
                         x=sel_var, y="avg_price_per_room",
                         color="booking_status",
                         color_discrete_map={"Not_Canceled":C_OK,"Canceled":C_CANCEL},
                         opacity=.5, marginal_x="histogram", marginal_y="histogram",
                         labels={"booking_status":"Estado"})
    fig_sc.update_traces(marker_size=4)
    apply_theme(fig_sc, f"{sel_var} vs Precio — por estado de reserva")
    st.plotly_chart(fig_sc, use_container_width=True)

# ────────────────────────────────────────────────────────────────────────────────
# TAB 5 — CHI² & GRANGER
# ────────────────────────────────────────────────────────────────────────────────
with tabs[4]:
    col_chi, col_gr = st.columns([1,1])

    # ── Chi-cuadrado
    with col_chi:
        st.markdown('<div class="section-header">Prueba Chi² — variables categóricas</div>', unsafe_allow_html=True)
        cat_vars = ["type_of_meal_plan","required_car_parking_space",
                    "repeated_guest","arrival_season","lead_time_cat","total_guests"]
        chi_rows = []
        for var in cat_vars:
            ct = pd.crosstab(df[var], df["booking_status"])
            if ct.shape[0] < 2 or ct.shape[1] < 2:
                continue
            chi2, p, dof, _ = chi2_contingency(ct)
            n  = ct.sum().sum()
            V  = np.sqrt(chi2 / (n * (min(ct.shape) - 1)))
            chi_rows.append({"Variable":str(var),"χ²":round(chi2,2),"p-valor":round(p,6),
                              "GL":dof,"Cramér's V":round(V,4),
                              "Sig":"✅" if p<0.05 else "❌"})
        df_chi = pd.DataFrame(chi_rows).sort_values("Cramér's V", ascending=False)

        fig_chi = px.bar(df_chi, x="Cramér's V", y="Variable",
                          orientation="h",
                          color="Cramér's V",
                          color_continuous_scale=[[0,C_G3],[.35,C_G5],[1,C_CANCEL]],
                          text=df_chi["Cramér's V"].apply(lambda v:f"{v:.3f}"),
                          labels={"Cramér's V":"Cramér's V"})
        fig_chi.update_traces(textposition="outside", marker_line_width=0)
        fig_chi.update_coloraxes(showscale=False)
        apply_theme(fig_chi, "Cramér's V — Asociación con booking_status")
        fig_chi.update_layout(yaxis_title="")
        st.plotly_chart(fig_chi, use_container_width=True)
        st.dataframe(df_chi, use_container_width=True, hide_index=True)

    # ── Granger
    with col_gr:
        st.markdown('<div class="section-header">Causalidad de Granger — series temporales</div>', unsafe_allow_html=True)
        df_full2 = df_full.copy()
        df_full2["date"] = pd.to_datetime(
            df_full2[["arrival_year","arrival_month","arrival_date"]]
            .rename(columns={"arrival_year":"year","arrival_month":"month","arrival_date":"day"}),
            errors="coerce")
        monthly_ts = df_full2.groupby(df_full2["date"].dt.to_period("M")).agg(
            cancel_rate=("canceled","mean"),
            avg_lead=("lead_time","mean"),
            avg_price=("avg_price_per_room","mean"),
            avg_requests=("no_of_special_requests","mean")
        ).reset_index()
        monthly_ts["date"] = monthly_ts["date"].astype(str)
        monthly_ts = monthly_ts.sort_values("date").reset_index(drop=True)

        # Series plot
        fig_ts = go.Figure()
        fig_ts.add_trace(go.Scatter(
            x=monthly_ts["date"], y=monthly_ts["cancel_rate"]*100,
            name="Tasa cancelación (%)", mode="lines+markers",
            line=dict(color=C_CANCEL,width=2),
            marker=dict(size=7)))
        fig_ts.add_trace(go.Scatter(
            x=monthly_ts["date"],
            y=(monthly_ts["avg_lead"]-monthly_ts["avg_lead"].mean())/monthly_ts["avg_lead"].std()*5+monthly_ts["cancel_rate"].mean()*100,
            name="Lead time (estand.)", mode="lines+markers",
            line=dict(color=C_OK,width=2,dash="dot"),
            marker=dict(size=5)))
        apply_theme(fig_ts, "Tasa de cancelación mensual vs Lead time (escalado)")
        fig_ts.update_layout(xaxis_title="Mes", yaxis_title="% / estand.",
                              xaxis_tickangle=-45)
        st.plotly_chart(fig_ts, use_container_width=True)

        max_lag = 3
        gr_rows = []
        for pred, label in [("avg_lead","Lead time"),("avg_price","Precio"),("avg_requests","Solicitudes")]:
            data_g = monthly_ts[["cancel_rate",pred]].dropna().values
            if len(data_g) < max_lag + 4:
                continue
            res = grangercausalitytests(data_g, maxlag=max_lag, verbose=False)
            for lag in range(1, max_lag+1):
                f  = res[lag][0]["ssr_ftest"][0]
                p  = res[lag][0]["ssr_ftest"][1]
                gr_rows.append({"Variable":label,"Lag":lag,
                                 "F":round(f,3),"p-valor":round(p,4),
                                 "Causalidad":"✅" if p<0.05 else "—"})
        df_gr = pd.DataFrame(gr_rows)
        if not df_gr.empty:
            pivot_gr = df_gr.pivot(index="Variable", columns="Lag", values="p-valor")
            fig_gr = px.imshow(pivot_gr, text_auto=".3f",
                                color_continuous_scale=[[0,C_G3],[.05,C_G5],[.2,C_CANCEL],[1,"#3D1A1A"]],
                                zmin=0, zmax=.2, aspect="auto",
                                labels=dict(x="Lag", color="p-valor"))
            apply_theme(fig_gr, "Granger — p-valores (verde < 0.05 = causalidad)")
            st.plotly_chart(fig_gr, use_container_width=True)
            st.dataframe(df_gr, use_container_width=True, hide_index=True)

    # ── Tablas de contingencia visuales
    st.markdown('<div class="section-header">Tablas de contingencia — Top 3 variables (Chi²)</div>', unsafe_allow_html=True)
    top3_vars = df_chi.head(3)["Variable"].tolist()
    cols_ct = st.columns(3)
    for col_ct, var in zip(cols_ct, top3_vars):
        ct_norm = pd.crosstab(df[var], df["booking_status"], normalize="index") * 100
        fig_ct = px.bar(ct_norm.reset_index(), x=var,
                         y=["Canceled","Not_Canceled"],
                         color_discrete_map={"Canceled":C_CANCEL,"Not_Canceled":C_OK},
                         barmode="stack",
                         labels={"value":"% dentro del grupo","variable":"Estado"})
        fig_ct.update_traces(marker_line_width=0)
        apply_theme(fig_ct, str(var))
        fig_ct.update_layout(height=320, yaxis_title="%", xaxis_title="",
                              legend_title="Estado", xaxis_tickangle=-25)
        col_ct.plotly_chart(fig_ct, use_container_width=True)

# ────────────────────────────────────────────────────────────────────────────────
# TAB 6 — EXPLORADOR
# ────────────────────────────────────────────────────────────────────────────────
with tabs[5]:
    st.markdown('<div class="section-header">Explorador de variables</div>', unsafe_allow_html=True)
    num_all = [c for c in df.select_dtypes(include=["int64","float64"]).columns
               if c not in ["canceled","meal_enc","season_enc"]]
    cat_all = ["type_of_meal_plan","arrival_season","lead_time_cat","booking_status",
               "repeated_guest","required_car_parking_space"]

    ex_col1, ex_col2, ex_col3 = st.columns(3)
    with ex_col1:
        x_var = st.selectbox("Eje X", num_all, index=num_all.index("lead_time"))
    with ex_col2:
        y_var = st.selectbox("Eje Y", num_all, index=num_all.index("avg_price_per_room"))
    with ex_col3:
        color_var = st.selectbox("Color", cat_all, index=0)

    chart_type = st.radio("Tipo de gráfico", ["Scatter","Box","Violin","Histogram"],
                           horizontal=True)
    n_show = st.slider("Registros a mostrar", 200, min(5000, len(df)), 1500, 100)

    df_ex = df.sample(n=min(n_show, len(df)), random_state=7)

    if chart_type == "Scatter":
        fig_ex = px.scatter(df_ex, x=x_var, y=y_var, color=color_var,
                             color_discrete_sequence=PALETTE, opacity=.6,
                             trendline="lowess" if len(df_ex) < 2000 else None,
                             hover_data=["booking_status","lead_time","avg_price_per_room"])
    elif chart_type == "Box":
        fig_ex = px.box(df_ex, x=color_var, y=y_var, color=color_var,
                         color_discrete_sequence=PALETTE, points="outliers")
    elif chart_type == "Violin":
        fig_ex = px.violin(df_ex, x=color_var, y=y_var, color=color_var,
                            color_discrete_sequence=PALETTE, box=True, points="outliers")
    else:
        fig_ex = px.histogram(df_ex, x=x_var, color=color_var,
                               color_discrete_sequence=PALETTE,
                               barmode="overlay", opacity=.7, nbins=40)

    fig_ex.update_traces(marker_line_width=0)
    apply_theme(fig_ex, f"{x_var} / {y_var} — por {color_var}")
    st.plotly_chart(fig_ex, use_container_width=True)

    # ── Estadísticas descriptivas por grupo
    st.markdown('<div class="section-header">Estadísticas descriptivas por grupo</div>', unsafe_allow_html=True)
    group_var = st.selectbox("Agrupar por", cat_all)
    desc = df.groupby(group_var, observed=True)[num_all].describe().round(2)
    st.dataframe(desc, use_container_width=True)

    # ── Descarga
    st.markdown('<div class="section-header">Descarga de datos filtrados</div>', unsafe_allow_html=True)
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Descargar dataset filtrado (.csv)",
                        data=csv, file_name="hotel_filtrado.csv",
                        mime="text/csv")

# ─── FOOTER ─────────────────────────────────────────────────────────────────────
st.markdown("""
<hr style='border-color:#30363D;margin:2rem 0 1rem'/>
<div style='text-align:center;color:#8B949E;font-size:.75rem;padding-bottom:1rem'>
  Hotel Bookings Analytics · Analítica y Visualización de Datos · Alejandro López<br>
  PCA · t-SNE · LLE · Mapa de Sammon · Pearson · Spearman · Kendall · Chi² · Granger
</div>
""", unsafe_allow_html=True)
