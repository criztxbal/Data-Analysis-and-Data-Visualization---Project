"""
Hotel Bookings — Dashboard de Analítica y Visualización
Alejandro López | Entrega Individual
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE, LocallyLinearEmbedding
from sklearn.metrics import pairwise_distances
from scipy.stats import pearsonr, spearmanr, chi2_contingency
from scipy.optimize import minimize
from sklearn.ensemble import RandomForestClassifier

# ── PAGE CONFIG ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Hotel Analytics · Alejandro López",
    page_icon="https://cdn-icons-png.flaticon.com/512/235/235889.png",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── ESTILOS (TEMA LIGHT PREMIUM / BLANCO HUESO CÁLIDO) ───────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&display=swap');

:root {
    --bg:   #F4F1EC;     /* hueso cálido, tira a arena */
    --surf: #FDFAF5;     /* blanco crema para tarjetas */
    --bord: #DDD8CE;     /* borde arena tenue */
    --acc:  #3B6FB5;     /* azul acero calmado */
    --txt:  #1C1C1E;     /* casi negro cálido */
    --muted:#5C5549;     /* gris pardo para subtítulos */
    --r: 12px;
}
html, body, [data-testid="stAppViewContainer"], [data-testid="stApp"] {
    background: var(--bg) !important;
    color: var(--txt);
    font-family: 'Inter', sans-serif !important;
}
[data-testid="stSidebar"] {
    background: var(--surf) !important;
    border-right: 1px solid var(--bord);
}
[data-testid="stSidebar"] * {
    color: var(--txt) !important;
}

/* Labels generales */
.stSelectbox label, .stSlider label,
.stMultiSelect label, .stRadio label {
    color: var(--txt) !important;
    font-weight: 800 !important;
    font-size: 0.95rem !important;
}

/* ── MULTISELECT — tags (chips) ─────────────────────────────────────── */
/* Contenedor del input */
[data-baseweb="select"] > div {
    background-color: var(--surf) !important;
    border-color: var(--bord) !important;
    border-radius: 8px !important;
}
/* Cada tag/chip seleccionado */
[data-baseweb="tag"] {
    background-color: rgba(59,111,181,0.12) !important;
    border: 1px solid rgba(59,111,181,0.30) !important;
    border-radius: 6px !important;
}
/* Texto dentro del tag */
[data-baseweb="tag"] span {
    color: #3B6FB5 !important;
    font-weight: 700 !important;
}
/* Ícono X dentro del tag */
[data-baseweb="tag"] [role="presentation"] svg {
    fill: #3B6FB5 !important;
}
/* Dropdown de opciones */
[data-baseweb="menu"] {
    background-color: var(--surf) !important;
    border: 1px solid var(--bord) !important;
    border-radius: 8px !important;
}
[data-baseweb="menu"] li {
    color: var(--txt) !important;
}
[data-baseweb="menu"] li:hover {
    background-color: rgba(59,111,181,0.08) !important;
}

/* ── PILLS (st.pills / radio como pastillas) ────────────────────────── */
/* Pastilla NO seleccionada */
[data-testid="stPillsContainer"] button {
    background: var(--surf) !important;
    border: 1.5px solid var(--bord) !important;
    color: var(--muted) !important;
    font-weight: 700 !important;
    border-radius: 999px !important;
    transition: all .15s ease;
}
[data-testid="stPillsContainer"] button:hover {
    border-color: #3B6FB5 !important;
    color: #3B6FB5 !important;
}
/* Pastilla SELECCIONADA */
[data-testid="stPillsContainer"] button[aria-pressed="true"],
[data-testid="stPillsContainer"] button[aria-checked="true"] {
    background: rgba(59,111,181,0.14) !important;
    border-color: #3B6FB5 !important;
    color: #3B6FB5 !important;
}

/* ── SLIDER ─────────────────────────────────────────────────────────── */
/* Track activo (la parte coloreada) */
[data-testid="stSlider"] [role="slider"] {
    background: #3B6FB5 !important;
    border-color: #3B6FB5 !important;
}
div[data-testid="stSlider"] > div > div > div > div {
    background: #3B6FB5 !important;
}
/* Número encima del thumb */
[data-testid="stSlider"] [data-testid="stTickBarMin"],
[data-testid="stSlider"] [data-testid="stTickBarMax"] {
    color: var(--muted) !important;
}

/* ── SELECTBOX dropdown ──────────────────────────────────────────────── */
[data-testid="stSelectbox"] > div > div {
    background-color: var(--surf) !important;
    border-color: var(--bord) !important;
    border-radius: 8px !important;
    color: var(--txt) !important;
}

/* Tarjetas de KPIs */
[data-testid="stMetric"] {
    background: var(--surf) !important;
    border: 1px solid var(--bord);
    border-radius: var(--r);
    padding: 1.2rem 1.5rem !important;
    box-shadow: 0 4px 6px rgba(0,0,0,0.04);
}
[data-testid="stMetricLabel"] {
    color: var(--muted) !important;
    font-size: 0.8rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    font-weight: 800;
}
[data-testid="stMetricValue"] {
    color: var(--txt) !important;
    font-size: 2.2rem;
    font-weight: 800;
}

/* Pestañas (Tabs) */
button[data-baseweb="tab"] {
    background: transparent !important;
    color: var(--muted) !important;
    border-bottom: 3px solid transparent !important;
    font-size: 0.9rem;
    font-weight: 800;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    padding-bottom: 0.5rem !important;
}
button[data-baseweb="tab"]:hover {
    color: var(--txt) !important;
}
button[data-baseweb="tab"][aria-selected="true"] {
    color: var(--acc) !important;
    border-bottom-color: var(--acc) !important;
    background: linear-gradient(0deg, rgba(59,111,181,0.06) 0%, transparent 100%) !important;
}

/* Títulos de sección */
.sh {
    font-size: 1rem;
    font-weight: 800;
    color: var(--txt);
    text-transform: uppercase;
    letter-spacing: 0.1em;
    border-left: 4px solid var(--acc);
    padding-left: 0.7rem;
    margin: 2rem 0 1rem;
    background: linear-gradient(90deg, rgba(59,111,181,0.06) 0%, transparent 100%);
    padding-top: 0.3rem;
    padding-bottom: 0.3rem;
}
.badge {
    display: inline-block;
    font-size: 0.7rem;
    font-weight: 800;
    padding: 0.25rem 0.75rem;
    border-radius: 999px;
    background: rgba(59,111,181,0.10);
    color: var(--acc);
    border: 1px solid rgba(59,111,181,0.25);
    margin-bottom: 0.5rem;
    letter-spacing: 0.05em;
}
</style>
""", unsafe_allow_html=True)

# ── COLORES & TEMA (GRÁFICAS) — paleta cálida armónica ───────────────────────
C_OK      = "#3B7DD8"   # azul índigo suave
C_CANCEL  = "#C0392B"   # rojo ladrillo oscuro
C_G3      = "#27AE60"   # verde salvia
C_G4      = "#7B68B5"   # violeta apagado
C_G5      = "#C87941"   # ámbar tostado
PAL = [C_OK, C_CANCEL, C_G3, C_G4, C_G5, "#2980B9", "#A93226"]

# Configuración base para fondo cálido
BASE = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color="#1C1C1E", family="Inter", size=13, weight="bold"),
    margin=dict(l=10, r=10, t=50, b=30),
    legend=dict(
        bgcolor="rgba(253,250,245,0.95)",
        bordercolor="#DDD8CE",
        borderwidth=1,
        font=dict(color="#1C1C1E", size=12, weight="bold")
    ),
    xaxis=dict(
        gridcolor="#E8E4DC",
        linecolor="#C4BFB5",
        zerolinecolor="#C4BFB5",
        title_font=dict(color="#1C1C1E", size=14, weight="bold"),
        tickfont=dict(color="#5C5549", size=12, weight="bold")
    ),
    yaxis=dict(
        gridcolor="#E8E4DC",
        linecolor="#C4BFB5",
        zerolinecolor="#C4BFB5",
        title_font=dict(color="#1C1C1E", size=14, weight="bold"),
        tickfont=dict(color="#5C5549", size=12, weight="bold")
    )
)

def T(fig, title=""):
    fig.update_layout(**BASE)
    if title:
        fig.update_layout(title=dict(
            text=title,
            font=dict(size=16, color="#1C1C1E", weight="bold"),
            x=0
        ))
    return fig

def sh(text):
    st.markdown(f'<div class="sh">{text}</div>', unsafe_allow_html=True)

MONTHS = {1:"Ene",2:"Feb",3:"Mar",4:"Abr",5:"May",6:"Jun",
          7:"Jul",8:"Ago",9:"Sep",10:"Oct",11:"Nov",12:"Dic"}

# ── CARGA & FEATURES ──────────────────────────────────────────────────────────
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
                                   labels=["<1 sem","1sem-1mes","1-3 meses",">3 meses"])
    le = LabelEncoder()
    df["meal_enc"]   = le.fit_transform(df["type_of_meal_plan"])
    df["season_enc"] = le.fit_transform(df["arrival_season"].astype(str))
    return df

df_full = load_data()

# ── SIDEBAR ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center;padding:1rem 0 .5rem'>
      <div style='font-size:2rem'><img src="https://cdn-icons-png.flaticon.com/512/235/235889.png"></div>
      <div style='font-weight:800;font-size:1.05rem;color:#1C1C1E;'>Hotel Analytics</div>
      <div style='font-size:.72rem;color:#5C5549'>Alejandro López</div>
    </div>
    <hr style='border-color:#DDD8CE;margin:.5rem 0 1rem'/>
    """, unsafe_allow_html=True)

    st.markdown('<div class="sh">Filtros globales</div>', unsafe_allow_html=True)
    sel_year   = st.selectbox("Año", ["Todos"] + [str(y) for y in sorted(df_full["arrival_year"].unique())])
    sel_months = st.multiselect("🗓 Meses", options=sorted(df_full["arrival_month"].unique()),
                                 default=sorted(df_full["arrival_month"].unique()),
                                 format_func=lambda m: MONTHS[m])
    sel_meal   = st.multiselect("Plan de comida",
                                 options=df_full["type_of_meal_plan"].unique().tolist(),
                                 default=df_full["type_of_meal_plan"].unique().tolist())
    sel_status = st.pills("Estado reserva",
                           options=["Todos","Not_Canceled","Canceled"], default="Todos")

    st.markdown('<hr style="border-color:#DDD8CE;margin:1rem 0"/>', unsafe_allow_html=True)
    st.markdown('<div class="sh">Proyecciones</div>', unsafe_allow_html=True)
    proj_n = st.slider("Muestra (t-SNE / LLE)", 300, 2000, 800, 100,
                        help="Reduce para acelerar el cálculo")

    st.markdown('<hr style="border-color:#DDD8CE;margin:1rem 0"/>', unsafe_allow_html=True)
    st.caption("Analítica y Visualización de Datos\nDataset: 25,811 reservas · 16 variables")

# ── FILTROS ───────────────────────────────────────────────────────────────────
df = df_full.copy()
if sel_year != "Todos":       df = df[df["arrival_year"] == int(sel_year)]
if sel_months:                df = df[df["arrival_month"].isin(sel_months)]
if sel_meal:                  df = df[df["type_of_meal_plan"].isin(sel_meal)]
if sel_status != "Todos":     df = df[df["booking_status"] == sel_status]

if df.empty:
    st.warning("Sin datos con los filtros actuales.")
    st.stop()

# ── HEADER ────────────────────────────────────────────────────────────────────
st.markdown(f"""
<div style='padding:.4rem 0 1.2rem'>
  <span class="badge">DASHBOARD INTERACTIVO</span>
  <h1 style='margin:0;font-size:1.9rem;font-weight:800;color:#1C1C1E;'>
    Hotel Bookings — Analítica Completa
  </h1>
  <p style='color:#5C5549;margin:.25rem 0 0;font-size:.88rem;font-weight:600;'>
    {len(df):,} reservas seleccionadas · {df["canceled"].mean()*100:.1f}% tasa de cancelación
  </p>
</div>""", unsafe_allow_html=True)

# ── KPIs ─────────────────────────────────────────────────────────────────────
k1,k2,k3,k4,k5 = st.columns(5)
k1.metric("Total reservas",     f"{len(df):,}",         f"{(len(df)/len(df_full)-1)*100:+.1f}% vs total")
k2.metric("Cancelaciones",      f"{df['canceled'].sum():,}", f"{df['canceled'].mean()*100:.1f}% tasa")
k3.metric("Precio promedio",    f"${df['avg_price_per_room'].mean():.0f}", f"σ=${df['avg_price_per_room'].std():.0f}")
k4.metric("Lead time promedio", f"{df['lead_time'].mean():.0f} días",     f"med {df['lead_time'].median():.0f}d")
k5.metric("Ingreso estimado",   f"${df['revenue_est'].sum()/1e6:.2f}M",   f"avg ${df['revenue_est'].mean():.0f}/res")
st.markdown("<div style='height:.5rem'/>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# TABS
# ══════════════════════════════════════════════════════════════════════════════
t1,t2,t3,t4,t5 = st.tabs([
    "Visión General",
    "Proyecciones",
    "Correlaciones",
    "Explorador",
    "Simulador"
])

# ──────────────────────────────────────────────────────────────────────────────
# TAB 1 — VISIÓN GENERAL
# ──────────────────────────────────────────────────────────────────────────────
with t1:
    c1, c2 = st.columns(2)

    with c1:
        sh("Estado de reservas")
        cnt = df["booking_status"].value_counts().reset_index()
        cnt.columns = ["status","count"]
        fig = go.Figure(go.Pie(
            labels=cnt["status"], values=cnt["count"], hole=.60,
            marker=dict(colors=[C_OK, C_CANCEL], line=dict(color="#FDFAF5", width=2)),
            textinfo="percent+label", textfont=dict(size=13, color="#FFFFFF", weight="bold")))
        fig.update_layout(**BASE, showlegend=False,
            title=dict(text="Distribución de estado", font=dict(size=14, color="#1C1C1E", weight="bold"), x=0),
            annotations=[dict(text=f"<b>{len(df):,}</b><br>reservas",
                              x=.5, y=.5, showarrow=False, font=dict(size=15, color="#1C1C1E"))])
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        sh("Cancelación por plan de comida")
        mg = df.groupby("type_of_meal_plan", observed=True).agg(
            tasa=("canceled","mean")).reset_index().sort_values("tasa")
        mg["pct"] = mg["tasa"]*100
        fig = px.bar(mg, y="type_of_meal_plan", x="pct", orientation="h",
                     color="pct", color_continuous_scale=[[0,C_G3],[.5,C_G5],[1,C_CANCEL]],
                     text=mg["pct"].apply(lambda v:f"{v:.1f}%"),
                     labels={"pct":"% cancelación","type_of_meal_plan":"Plan"})
        fig.update_traces(textposition="outside", marker_line_width=0, textfont=dict(weight="bold"))
        fig.update_coloraxes(showscale=False)
        T(fig,"Tasa de cancelación (%)"); fig.update_layout(yaxis_title="")
        st.plotly_chart(fig, use_container_width=True)

    c3, c4 = st.columns(2)

    with c3:
        sh("Lead time vs cancelación")
        fig = go.Figure()
        for status, color, name in [("Not_Canceled", C_OK, "No cancelada"),
                                     ("Canceled", C_CANCEL, "Cancelada")]:
            s = df[df["booking_status"]==status]["lead_time"]
            fig.add_trace(go.Histogram(x=s, name=f"{name} (μ={s.mean():.0f}d)",
                nbinsx=40, opacity=.8, marker_color=color, marker_line_width=0))
        T(fig,"Distribución Lead Time")
        fig.update_layout(barmode="overlay", xaxis_title="Días", yaxis_title="Frecuencia")
        st.plotly_chart(fig, use_container_width=True)

    with c4:
        sh("Precio por habitación")
        df_b = df[df["avg_price_per_room"]>0].copy()
        fig = px.box(df_b, x="booking_status", y="avg_price_per_room",
                     color="booking_status", points="outliers",
                     color_discrete_map={"Not_Canceled":C_OK,"Canceled":C_CANCEL},
                     labels={"booking_status":"Estado","avg_price_per_room":"USD/noche"})
        fig.update_traces(marker_size=4, opacity=.9, line=dict(width=2))
        T(fig,"Precio por habitación"); fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    sh("Evaluación de outliers — método IQR")
    num_cols = df.select_dtypes(include=["int64","float64"]).columns.difference(
        ["canceled","meal_enc","season_enc","repeated_guest","required_car_parking_space"])
    rows = []
    for col in num_cols:
        Q1,Q3 = df[col].quantile([.25,.75])
        IQR = Q3-Q1
        n = ((df[col]<Q1-1.5*IQR)|(df[col]>Q3+1.5*IQR)).sum()
        rows.append({"Variable":col,"Q1":round(Q1,2),"Q3":round(Q3,2),
                     "IQR":round(IQR,2),"Outliers":n,"% total":round(n/len(df)*100,2)})
    dfo = pd.DataFrame(rows).sort_values("Outliers",ascending=False)
    fig = px.bar(dfo, x="Variable", y="% total", color="% total",
                 color_continuous_scale=[[0,C_G3],[.4,C_G5],[1,C_CANCEL]],
                 text=dfo["% total"].apply(lambda v:f"{v:.1f}%"),
                 labels={"% total":"% outliers"})
    fig.update_traces(textposition="outside", marker_line_width=0, textfont=dict(weight="bold"))
    fig.update_coloraxes(showscale=False)
    T(fig,"Porcentaje de outliers por variable")
    st.plotly_chart(fig, use_container_width=True)

# ──────────────────────────────────────────────────────────────────────────────
# TAB 2 — PROYECCIONES
# ──────────────────────────────────────────────────────────────────────────────
with t2:
    FEATURES = ["lead_time","avg_price_per_room","no_of_special_requests",
                "no_of_week_nights","no_of_weekend_nights","no_of_adults",
                "no_of_children","no_of_previous_cancellations",
                "no_of_previous_bookings_not_canceled",
                "total_nights","total_guests","meal_enc","season_enc",
                "required_car_parking_space"]

    @st.cache_data
    def calc_pca():
        X = df_full[FEATURES].values
        Xs = StandardScaler().fit_transform(X)
        pf = PCA(random_state=42).fit(Xs)
        p2 = PCA(n_components=2, random_state=42).fit(Xs)
        np.random.seed(42)
        idx = np.random.choice(len(Xs), 1000, replace=False)
        return (pf.explained_variance_ratio_,
                p2.explained_variance_ratio_,
                p2.components_,
                p2.transform(Xs[idx]),
                df_full["canceled"].values[idx])

    ev_full, ev2, comps, X_pca, y_pca = calc_pca()

    @st.cache_data(show_spinner=False)
    def calc_nonlinear(n: int):
        X = df_full[FEATURES].values
        Xs = StandardScaler().fit_transform(X)
        np.random.seed(42)
        idx = np.random.choice(len(Xs), size=min(n, len(Xs)), replace=False)
        Xs2 = Xs[idx]; y2 = df_full["canceled"].values[idx]
        Xt = TSNE(n_components=2, perplexity=30, init="pca",
                  learning_rate="auto", random_state=42, n_jobs=-1).fit_transform(Xs2)
        Xl = LocallyLinearEmbedding(n_components=2, n_neighbors=12,
                                     random_state=42, n_jobs=-1).fit_transform(Xs2)
        ns = min(400, len(Xs2))
        Ds = pairwise_distances(Xs2[:ns]); Ds[Ds==0] = 1e-10; cs = Ds.sum()
        def stress(flat, D, n, c):
            C = flat.reshape(n,2); Dl = pairwise_distances(C); Dl[Dl==0]=1e-10
            return (1/c)*np.sum((D-Dl)**2/D)
        x0 = PCA(n_components=2,random_state=42).fit_transform(Xs2[:ns]).flatten()
        res = minimize(stress, x0, args=(Ds,ns,cs), method="L-BFGS-B",
                       options={"maxiter":250,"ftol":1e-6})
        Xsm = res.x.reshape(ns,2)
        return Xt, Xl, Xsm, y2, y2[:ns]

    pa, pb = st.columns(2)
    with pa:
        sh("PCA — Scree Plot (varianza explicada)")
        cv = np.cumsum(ev_full)*100
        fig = go.Figure()
        fig.add_trace(go.Bar(x=list(range(1,len(ev_full)+1)), y=ev_full*100,
                              name="Por componente", marker_color=C_OK,
                              marker_line_width=0, opacity=.9))
        fig.add_trace(go.Scatter(x=list(range(1,len(ev_full)+1)), y=cv,
                                  name="Acumulada", mode="lines+markers",
                                  line=dict(color=C_CANCEL,width=3),
                                  marker=dict(size=7)))
        fig.add_hline(y=80, line_dash="dash", line_color="#5C5549",
                      annotation_text="80%", annotation_font_color="#5C5549")
        T(fig,"Varianza por componente")
        fig.update_layout(xaxis_title="Componente", yaxis_title="%", yaxis_range=[0,107])
        st.plotly_chart(fig, use_container_width=True)

    with pb:
        sh("PCA — Loadings PC1 vs PC2")
        ld = pd.DataFrame(comps.T, index=FEATURES, columns=["PC1","PC2"])
        ld = ld.reindex(ld["PC1"].abs().sort_values(ascending=False).index)
        fig = go.Figure()
        fig.add_trace(go.Bar(x=ld.index, y=ld["PC1"], name="PC1",
                              marker_color=C_OK, opacity=.9, marker_line_width=0))
        fig.add_trace(go.Bar(x=ld.index, y=ld["PC2"], name="PC2",
                              marker_color=C_G4, opacity=.9, marker_line_width=0))
        T(fig,"Contribución a PC1 y PC2")
        fig.update_layout(barmode="group", xaxis_tickangle=-38,
                           xaxis_title="", yaxis_title="Loading")
        st.plotly_chart(fig, use_container_width=True)

    sh("PCA — Biplot 2D (1,000 puntos)")
    df_pca_plot = pd.DataFrame({"x":X_pca[:,0],"y":X_pca[:,1],
        "clase":["Cancelada" if v else "No cancelada" for v in y_pca]})
    fig = px.scatter(df_pca_plot, x="x", y="y", color="clase",
                     color_discrete_map={"No cancelada":C_OK,"Cancelada":C_CANCEL},
                     opacity=.6, labels={"x":f"PC1 ({ev2[0]*100:.1f}% var)",
                                          "y":f"PC2 ({ev2[1]*100:.1f}% var)"})
    sc = 3.5
    for i, feat in enumerate(FEATURES):
        fig.add_annotation(
            x=comps[0,i]*sc, y=comps[1,i]*sc, ax=0, ay=0,
            text=feat, font=dict(size=11, color="#1C1C1E", weight="bold"),
            bgcolor="rgba(253,250,245,0.92)", bordercolor="#3B6FB5", borderwidth=1.5, borderpad=3
        )
    fig.update_traces(marker_size=5)
    T(fig,"PCA Biplot — variables y observaciones")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("<div style='height:.5rem'/>", unsafe_allow_html=True)
    sh("Proyecciones no lineales — t-SNE · LLE · Sammon")
    st.info(f"⏱ Calcular con **{proj_n}** puntos (~15-40 s). El resultado se cachea automáticamente.")

    if st.button("Calcular t-SNE, LLE y Mapa de Sammon", type="primary"):
        st.session_state["run_proj"] = True

    if st.session_state.get("run_proj"):
        with st.spinner("Calculando proyecciones no lineales…"):
            Xt, Xl, Xsm, ys, ys_sm = calc_nonlinear(proj_n)

        def proj_fig(X2d, yy, title):
            dp = pd.DataFrame({"x":X2d[:,0],"y":X2d[:,1],
                "clase":["Cancelada" if v else "No cancelada" for v in yy]})
            f = px.scatter(dp, x="x", y="y", color="clase",
                           color_discrete_map={"No cancelada":C_OK,"Cancelada":C_CANCEL},
                           opacity=.6)
            f.update_traces(marker_size=4)
            T(f, title); f.update_layout(height=370, xaxis_title="Dim 1", yaxis_title="Dim 2")
            return f

        pc1, pc2, pc3 = st.columns(3)
        with pc1: st.plotly_chart(proj_fig(Xt,  ys,     "t-SNE"),         use_container_width=True)
        with pc2: st.plotly_chart(proj_fig(Xl,  ys,     "LLE"),           use_container_width=True)
        with pc3: st.plotly_chart(proj_fig(Xsm, ys_sm,  "Sammon (n=400)"), use_container_width=True)

        st.success("Proyecciones completadas. PCA (lineal) muestra varianza global. t-SNE y LLE preservan estructura local. Sammon minimiza el estrés de distancias.")

# ──────────────────────────────────────────────────────────────────────────────
# TAB 3 — CORRELACIONES
# ──────────────────────────────────────────────────────────────────────────────
with t3:
    NUM_VARS = ["lead_time","avg_price_per_room","no_of_special_requests",
                "no_of_week_nights","no_of_weekend_nights","no_of_adults",
                "no_of_previous_cancellations","no_of_previous_bookings_not_canceled",
                "total_nights","total_guests","canceled"]

    m_col, s_col = st.columns(2)

    with m_col:
        sh("Método de correlación")
        corr_method = st.radio("",["pearson","spearman"], horizontal=True,
            format_func=lambda m:{"pearson":"Pearson r","spearman":"Spearman ρ"}[m],
            key="corr_radio")
        cm = df[NUM_VARS].corr(method=corr_method)
        mask = np.triu(np.ones_like(cm, dtype=bool), k=1)
        cd = cm.where(~mask)
        fig = px.imshow(cd, text_auto=".2f", color_continuous_scale="RdBu_r",
                        zmin=-1, zmax=1, aspect="auto", color_continuous_midpoint=0,
                        labels=dict(color="r/ρ/τ"))
        T(fig, f"Matriz de correlación — {corr_method.capitalize()}")
        fig.update_traces(textfont_size=10)
        st.plotly_chart(fig, use_container_width=True)

    with s_col:
        sh("Coeficientes vs 'canceled'")
        res_list = []
        for col in NUM_VARS:
            if col == "canceled": continue
            rp,pp = pearsonr(df[col], df["canceled"])
            rs,ps = spearmanr(df[col], df["canceled"])
            res_list.append({"Variable":col,
                "Pearson r":round(rp,4),"p_p":round(pp,5),
                "Spearman ρ":round(rs,4),"p_s":round(ps,5)})
        df_cr = pd.DataFrame(res_list).sort_values("Pearson r", key=abs, ascending=False)

        fig = go.Figure()
        xv = df_cr["Variable"]
        for col_n, color, name in [("Pearson r", C_OK, "Pearson r"),
                                    ("Spearman ρ", C_G4, "Spearman ρ")]:
            fig.add_trace(go.Bar(name=name, x=xv, y=df_cr[col_n],
                                  marker_color=color, opacity=.9, marker_line_width=0))
        fig.add_hline(y=0, line_color="#DDD8CE")
        T(fig,"Correlación con 'canceled'")
        fig.update_layout(barmode="group", xaxis_tickangle=-38,
                           yaxis_title="Coeficiente", xaxis_title="")
        st.plotly_chart(fig, use_container_width=True)

    sh("Tabla de coeficientes y significancia")
    df_show = df_cr.copy()
    df_show["Sig."] = df_show["p_p"].apply(lambda p:"Correcto" if p<0.05 else "Incorrecto")
    st.dataframe(df_show.rename(columns={"p_p":"p (P)","p_s":"p (S)"}),
                 use_container_width=True, hide_index=True)

    sh("Scatter interactivo — variable seleccionada vs precio")
    sel_v = st.selectbox("Variable X:", [v for v in NUM_VARS if v!="canceled"],
                          key="scatter_var")
    fig = px.scatter(df.sample(min(2000,len(df)), random_state=1),
                     x=sel_v, y="avg_price_per_room", color="booking_status",
                     color_discrete_map={"Not_Canceled":C_OK,"Canceled":C_CANCEL},
                     opacity=.6, marginal_x="histogram", marginal_y="histogram",
                     labels={"booking_status":"Estado"})
    fig.update_traces(selector=dict(type="scatter"), marker_size=5)
    T(fig, f"{sel_v} vs Precio — por estado")
    st.plotly_chart(fig, use_container_width=True)

# ──────────────────────────────────────────────────────────────────────────────
# TAB 4 — EXPLORADOR
# ──────────────────────────────────────────────────────────────────────────────
with t4:
    num_all = [c for c in df_full.select_dtypes(include=["int64","float64"]).columns
               if c not in ["canceled","meal_enc","season_enc"]]
    cat_all = ["type_of_meal_plan","arrival_season","lead_time_cat",
               "booking_status","repeated_guest","required_car_parking_space"]

    sh("Constructor de gráficos")
    ex1, ex2, ex3 = st.columns(3)
    with ex1:
        x_var = st.selectbox("Eje X", num_all,
            index=num_all.index("lead_time") if "lead_time" in num_all else 0,
            key="ex_x")
    with ex2:
        y_var = st.selectbox("Eje Y", num_all,
            index=num_all.index("avg_price_per_room") if "avg_price_per_room" in num_all else 1,
            key="ex_y")
    with ex3:
        color_var = st.selectbox("Color (categoría)", cat_all, key="ex_color")

    chart_type = st.radio("Tipo de gráfico",
                           ["Scatter","Box","Violin","Histogram"],
                           horizontal=True, key="ex_chart")
    n_show = st.slider("Puntos a mostrar", 200, min(4000,len(df)), 1500, 100, key="ex_n")

    df_ex = df.sample(n=min(n_show,len(df)), random_state=7)

    if chart_type == "Scatter":
        fig_ex = px.scatter(df_ex, x=x_var, y=y_var, color=color_var,
                             color_discrete_sequence=PAL, opacity=.65,
                             hover_data=["booking_status","lead_time","avg_price_per_room"])
    elif chart_type == "Box":
        fig_ex = px.box(df_ex, x=color_var, y=y_var, color=color_var,
                         color_discrete_sequence=PAL, points="outliers")
    elif chart_type == "Violin":
        fig_ex = px.violin(df_ex, x=color_var, y=y_var, color=color_var,
                            color_discrete_sequence=PAL, box=True)
    else:  # Histogram
        fig_ex = px.histogram(df_ex, x=x_var, color=color_var,
                               color_discrete_sequence=PAL,
                               barmode="overlay", opacity=.8, nbins=40)

    # ── CORRECCIÓN DEL BUG ────────────────────────────────────────────────────
    # marker_line_width se puede aplicar a todos los tipos sin problema.
    # marker_size solo se aplica a scatter (histogramas/box/violin no lo soportan).
    fig_ex.update_traces(marker_line_width=0)
    if chart_type == "Scatter":
        fig_ex.update_traces(marker_size=5, selector=dict(type="scatter"))
    # ─────────────────────────────────────────────────────────────────────────

    T(fig_ex, f"{x_var} · {y_var} — por {color_var}")
    st.plotly_chart(fig_ex, use_container_width=True)

    sh("Estadísticas descriptivas por grupo")
    group_var = st.selectbox("Agrupar por", cat_all, key="ex_group")
    desc_df = df.groupby(group_var, observed=True)[num_all].describe().round(2)
    st.dataframe(desc_df, use_container_width=True)

    sh("Descargar datos filtrados")
    csv_bytes = df.to_csv(index=False).encode("utf-8")
    st.download_button("Descargar dataset filtrado (.csv)", data=csv_bytes,
                        file_name="hotel_filtrado.csv", mime="text/csv")

# ──────────────────────────────────────────────────────────────────────────────
# TAB 5 — SIMULADOR Y DESGLOSE (PREDICTIVO ML)
# ──────────────────────────────────────────────────────────────────────────────
with t5:
    sh("Perfil de Cliente y Simulador de Cancelación")
    st.write("Selecciona un cliente de la base actual filtrada para ver el desglose de su información y evaluar la probabilidad de que cancele su reservación usando nuestro modelo de Machine Learning.")

    @st.cache_resource(show_spinner=False)
    def train_rf_model(df_train):
        features = ["lead_time", "avg_price_per_room", "no_of_special_requests",
                    "no_of_week_nights", "no_of_weekend_nights", "no_of_adults",
                    "total_nights", "total_guests"]
        X = df_train[features].fillna(0)
        y = df_train["canceled"]
        rf = RandomForestClassifier(n_estimators=50, max_depth=6, random_state=42)
        rf.fit(X, y)
        return rf, features

    rf_model, mod_features = train_rf_model(df_full)

    col_sim1, col_sim2 = st.columns([1, 2])

    with col_sim1:
        st.markdown("### Seleccionar Cliente")
        client_options = df.index.tolist()

        if not client_options:
            st.warning("No hay clientes bajo los filtros actuales.")
        else:
            selected_id = st.selectbox("Elige el ID de la reserva:", client_options[:500],
                                        help="Listando los primeros 500 bajo el filtro actual.")

            if selected_id is not None:
                client_data = df.loc[[selected_id]]
                X_client = client_data[mod_features].fillna(0)

                prob_cancel = rf_model.predict_proba(X_client)[0][1]
                pred_label = "Va a cancelar" if prob_cancel > 0.5 else "No va a cancelar"
                real_status = client_data['booking_status'].values[0]

                st.markdown("### Predicción del Modelo")
                st.metric(label="Resultado Predictivo", value=pred_label)
                st.metric(label="Probabilidad de Cancelación", value=f"{prob_cancel*100:.1f}%")
                st.caption(f"Status real en la base: **{real_status}**")

    with col_sim2:
        if client_options and selected_id is not None:
            st.markdown("### Desglose de Datos del Cliente")
            st.write("Información detallada utilizada para perfilar a esta reserva en particular:")

            disp_df = client_data[mod_features].T.rename(columns={selected_id: "Valor"})
            st.table(disp_df.style.format("{:.2f}", na_rep="-").set_properties(**{
                'background-color': '#FDFAF5',
                'color': '#1C1C1E',
                'border-color': '#DDD8CE',
                'font-weight': 'bold'
            }))

            st.markdown("### Comparativa de variables clave vs Promedio")

            df_means = df_full[mod_features].mean()
            client_vals = client_data[mod_features].iloc[0]

            comp_df = pd.DataFrame({
                "Métrica": mod_features,
                "Este Cliente": client_vals.values,
                "Promedio Global": df_means.values
            })

            fig_comp = px.bar(comp_df, x="Métrica", y=["Este Cliente", "Promedio Global"],
                              barmode="group",
                              color_discrete_map={"Este Cliente": C_OK, "Promedio Global": "#C4BFB5"})
            T(fig_comp, "Análisis de peso de variables del Cliente")
            fig_comp.update_layout(yaxis_title="Unidades / Días / $", xaxis_tickangle=-35)
            st.plotly_chart(fig_comp, use_container_width=True)

# ── FOOTER ────────────────────────────────────────────────────────────────────
st.markdown("""
<hr style='border-color:#DDD8CE;margin:2rem 0 .8rem'/>
<div style='text-align:center;color:#5C5549;font-size:.75rem;font-weight:600;padding-bottom:.8rem'>
  Hotel Bookings Analytics · Analítica y Visualización de Datos · Alejandro López<br>
</div>
""", unsafe_allow_html=True)
