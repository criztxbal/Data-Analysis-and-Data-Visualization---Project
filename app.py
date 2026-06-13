"""
Hotel Bookings — Dashboard de Analítica y Visualización
Alejandro López | Entrega Individual
"""

import warnings, logging
warnings.filterwarnings("ignore")
logging.getLogger("statsmodels").setLevel(logging.ERROR)

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE, LocallyLinearEmbedding
from sklearn.metrics import pairwise_distances
from scipy.stats import pearsonr, spearmanr, kendalltau, chi2_contingency
from statsmodels.tsa.stattools import grangercausalitytests
from scipy.optimize import minimize

# ── PAGE CONFIG ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Hotel Analytics · Alejandro López",
    page_icon="🏨",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── ESTILOS ───────────────────────────────────────────────────────────────────
st.markdown("""
<style>
:root{--bg:#0D1117;--surf:#161B22;--bord:#30363D;--acc:#58A6FF;
      --red:#F78166;--grn:#3FB950;--purp:#D2A8FF;--ora:#FFA657;
      --txt:#E6EDF3;--muted:#8B949E;--r:12px;}
html,body,[data-testid="stAppViewContainer"],[data-testid="stApp"]
  {background:var(--bg)!important;color:var(--txt);font-family:'Inter',sans-serif;}
[data-testid="stSidebar"]{background:var(--surf)!important;border-right:1px solid var(--bord);}
[data-testid="stSidebar"] *{color:var(--txt)!important;}
[data-testid="stMetric"]{background:var(--surf)!important;border:1px solid var(--bord);
  border-radius:var(--r);padding:1rem 1.25rem!important;}
[data-testid="stMetricLabel"]{color:var(--muted)!important;font-size:.72rem;
  text-transform:uppercase;letter-spacing:.06em;}
[data-testid="stMetricValue"]{color:var(--txt)!important;font-size:1.85rem;font-weight:700;}
button[data-baseweb="tab"]{background:transparent!important;color:var(--muted)!important;
  border-bottom:2px solid transparent!important;font-size:.82rem;font-weight:600;
  text-transform:uppercase;letter-spacing:.06em;}
button[data-baseweb="tab"][aria-selected="true"]{color:var(--acc)!important;
  border-bottom-color:var(--acc)!important;}
.sh{font-size:.95rem;font-weight:700;color:var(--muted);text-transform:uppercase;
    letter-spacing:.08em;border-left:3px solid var(--acc);padding-left:.55rem;
    margin:1.4rem 0 .6rem;}
.badge{display:inline-block;font-size:.7rem;font-weight:700;padding:.2rem .6rem;
  border-radius:999px;background:rgba(88,166,255,.15);color:var(--acc);
  border:1px solid rgba(88,166,255,.3);margin-bottom:.4rem;}
</style>
""", unsafe_allow_html=True)

# ── COLORES & TEMA ─────────────────────────────────────────────────────────────
C_OK, C_CANCEL, C_G3, C_G4, C_G5 = "#58A6FF","#F78166","#3FB950","#D2A8FF","#FFA657"
PAL = [C_OK, C_CANCEL, C_G3, C_G4, C_G5, "#79C0FF", "#FF7B72"]
BASE = dict(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#E6EDF3", family="Inter"),
            margin=dict(l=20,r=20,t=42,b=20),
            legend=dict(bgcolor="rgba(0,0,0,0)",bordercolor="#30363D",borderwidth=1),
            xaxis=dict(gridcolor="#21262D",linecolor="#30363D",zerolinecolor="#30363D"),
            yaxis=dict(gridcolor="#21262D",linecolor="#30363D",zerolinecolor="#30363D"))

def T(fig, title=""):
    fig.update_layout(**BASE)
    if title:
        fig.update_layout(title=dict(text=title,font=dict(size=13,color="#E6EDF3"),x=0))
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
      <div style='font-size:2rem'>🏨</div>
      <div style='font-weight:800;font-size:1.05rem'>Hotel Analytics</div>
      <div style='font-size:.72rem;color:#8B949E'>Alejandro López</div>
    </div>
    <hr style='border-color:#30363D;margin:.5rem 0 1rem'/>
    """, unsafe_allow_html=True)

    st.markdown('<div class="sh">Filtros globales</div>', unsafe_allow_html=True)
    sel_year   = st.selectbox("📅 Año", ["Todos"] + [str(y) for y in sorted(df_full["arrival_year"].unique())])
    sel_months = st.multiselect("🗓 Meses", options=sorted(df_full["arrival_month"].unique()),
                                 default=sorted(df_full["arrival_month"].unique()),
                                 format_func=lambda m: MONTHS[m])
    sel_meal   = st.multiselect("🍽 Plan de comida",
                                 options=df_full["type_of_meal_plan"].unique().tolist(),
                                 default=df_full["type_of_meal_plan"].unique().tolist())
    sel_status = st.pills("Estado reserva",
                           options=["Todos","Not_Canceled","Canceled"], default="Todos")

    st.markdown('<hr style="border-color:#30363D;margin:1rem 0"/>', unsafe_allow_html=True)
    st.markdown('<div class="sh">Proyecciones</div>', unsafe_allow_html=True)
    proj_n = st.slider("Muestra (t-SNE / LLE)", 300, 2000, 800, 100,
                        help="Reduce para acelerar el cálculo")

    st.markdown('<hr style="border-color:#30363D;margin:1rem 0"/>', unsafe_allow_html=True)
    st.caption("Analítica y Visualización de Datos\nDataset: 25,811 reservas · 16 variables")

# ── FILTROS ───────────────────────────────────────────────────────────────────
df = df_full.copy()
if sel_year != "Todos":       df = df[df["arrival_year"] == int(sel_year)]
if sel_months:                df = df[df["arrival_month"].isin(sel_months)]
if sel_meal:                  df = df[df["type_of_meal_plan"].isin(sel_meal)]
if sel_status != "Todos":     df = df[df["booking_status"] == sel_status]

if df.empty:
    st.warning("⚠️ Sin datos con los filtros actuales.")
    st.stop()

# ── HEADER ────────────────────────────────────────────────────────────────────
st.markdown(f"""
<div style='padding:.4rem 0 1.2rem'>
  <span class="badge">DASHBOARD INTERACTIVO</span>
  <h1 style='margin:0;font-size:1.9rem;font-weight:800;
    background:linear-gradient(90deg,#58A6FF,#D2A8FF);
    -webkit-background-clip:text;-webkit-text-fill-color:transparent'>
    Hotel Bookings — Analítica Completa
  </h1>
  <p style='color:#8B949E;margin:.25rem 0 0;font-size:.88rem'>
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
t1,t2,t3,t4,t5,t6 = st.tabs([
    "📊 Visión General",
    "📈 Series & Estacionalidad",
    "🔵 Proyecciones",
    "🔗 Correlaciones",
    "🧮 Chi² & Granger",
    "🔍 Explorador",
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
            marker=dict(colors=[C_OK,C_CANCEL], line=dict(color="#0D1117",width=3)),
            textinfo="percent+label", textfont=dict(size=13)))
        fig.update_layout(**BASE, showlegend=False,
            title=dict(text="Distribución de estado",font=dict(size=13,color="#E6EDF3"),x=0),
            annotations=[dict(text=f"<b>{len(df):,}</b><br>reservas",
                              x=.5,y=.5,showarrow=False,font=dict(size=15,color="#E6EDF3"))])
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
        fig.update_traces(textposition="outside", marker_line_width=0)
        fig.update_coloraxes(showscale=False)
        T(fig,"Tasa de cancelación (%)"); fig.update_layout(yaxis_title="")
        st.plotly_chart(fig, use_container_width=True)

    c3, c4 = st.columns(2)

    with c3:
        sh("Lead time vs cancelación")
        fig = go.Figure()
        for status, color, name in [("Not_Canceled",C_OK,"No cancelada"),
                                     ("Canceled",C_CANCEL,"Cancelada")]:
            s = df[df["booking_status"]==status]["lead_time"]
            fig.add_trace(go.Histogram(x=s, name=f"{name} (μ={s.mean():.0f}d)",
                nbinsx=40, opacity=.72, marker_color=color, marker_line_width=0))
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
        fig.update_traces(marker_size=3, opacity=.85)
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
    fig.update_traces(textposition="outside", marker_line_width=0)
    fig.update_coloraxes(showscale=False)
    T(fig,"Porcentaje de outliers por variable")
    st.plotly_chart(fig, use_container_width=True)

# ──────────────────────────────────────────────────────────────────────────────
# TAB 2 — SERIES & ESTACIONALIDAD
# ──────────────────────────────────────────────────────────────────────────────
with t2:
    sh("Volumen mensual de reservas")
    mly = df.groupby("arrival_month",observed=True).agg(
        total=("canceled","count"), canceladas=("canceled","sum")).reset_index()
    mly["no_c"] = mly["total"] - mly["canceladas"]
    mly["mes"]  = mly["arrival_month"].map(MONTHS)
    mly["tasa"] = mly["canceladas"]/mly["total"]*100

    fig = go.Figure()
    fig.add_trace(go.Bar(x=mly["mes"], y=mly["no_c"],   name="No cancelada",
                          marker_color=C_OK,     marker_line_width=0))
    fig.add_trace(go.Bar(x=mly["mes"], y=mly["canceladas"], name="Cancelada",
                          marker_color=C_CANCEL,  marker_line_width=0))
    T(fig,"Reservas por mes"); fig.update_layout(barmode="stack",
        yaxis_title="Reservas", xaxis_title="Mes")
    st.plotly_chart(fig, use_container_width=True)

    ca, cb = st.columns(2)
    with ca:
        sh("Tasa de cancelación mensual")
        fig = go.Figure(go.Scatter(
            x=mly["mes"], y=mly["tasa"].round(1),
            mode="lines+markers+text",
            line=dict(color=C_CANCEL,width=2.5),
            marker=dict(size=9,color=C_CANCEL,line=dict(color="#0D1117",width=2)),
            text=mly["tasa"].apply(lambda v:f"{v:.1f}%"), textposition="top center",
            textfont=dict(size=10)))
        T(fig,"% Cancelación por mes")
        fig.update_layout(yaxis_title="%", xaxis_title="Mes",
                          yaxis_range=[0,mly["tasa"].max()*1.35])
        st.plotly_chart(fig, use_container_width=True)

    with cb:
        sh("Ingreso estimado por temporada")
        sr = df[df["canceled"]==0].groupby("arrival_season",observed=True)["revenue_est"].mean().reset_index()
        sr.columns = ["Temporada","Promedio"]
        fig = px.bar(sr, x="Temporada", y="Promedio", color="Temporada",
                     color_discrete_sequence=PAL,
                     text=sr["Promedio"].apply(lambda v:f"${v:,.0f}"))
        fig.update_traces(textposition="outside", marker_line_width=0)
        T(fig,"Ingreso promedio por reserva confirmada (USD)")
        fig.update_layout(showlegend=False, yaxis_title="USD")
        st.plotly_chart(fig, use_container_width=True)

    sh("Heatmap — % Cancelación mes × plan de comida")
    ph = df.groupby(["arrival_month","type_of_meal_plan"],observed=True)["canceled"].mean().unstack()*100
    ph.index = [MONTHS[m] for m in ph.index]
    fig = px.imshow(ph, color_continuous_scale=[[0,C_G3],[.4,C_G5],[1,C_CANCEL]],
                    text_auto=".0f", aspect="auto",
                    labels=dict(x="Plan de comida",y="Mes",color="% Cancel."))
    T(fig,"Cancelación (%) — Mes × Plan de comida")
    fig.update_coloraxes(colorbar_title="% Cancel.")
    st.plotly_chart(fig, use_container_width=True)

    sh("Lead time — estado por categoría de anticipación")
    lt = df.groupby(["lead_time_cat","booking_status"],observed=True).size().reset_index(name="n")
    lt["pct"] = lt["n"] / lt.groupby("lead_time_cat",observed=True)["n"].transform("sum") * 100
    fig = px.bar(lt, x="lead_time_cat", y="pct", color="booking_status",
                 color_discrete_map={"Not_Canceled":C_OK,"Canceled":C_CANCEL},
                 barmode="stack", text=lt["pct"].apply(lambda v:f"{v:.0f}%"),
                 labels={"lead_time_cat":"Anticipación","pct":"% del grupo","booking_status":"Estado"})
    fig.update_traces(textposition="inside", textfont_size=11, marker_line_width=0)
    T(fig,"Estado según anticipación de reserva"); fig.update_layout(yaxis_title="%")
    st.plotly_chart(fig, use_container_width=True)

# ──────────────────────────────────────────────────────────────────────────────
# TAB 3 — PROYECCIONES  (lazy: solo carga al abrir la pestaña)
# ──────────────────────────────────────────────────────────────────────────────
with t3:
    FEATURES = ["lead_time","avg_price_per_room","no_of_special_requests",
                "no_of_week_nights","no_of_weekend_nights","no_of_adults",
                "no_of_children","no_of_previous_cancellations",
                "no_of_previous_bookings_not_canceled",
                "total_nights","total_guests","meal_enc","season_enc",
                "required_car_parking_space"]

    # PCA se calcula siempre (rápido)
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

    # t-SNE y LLE solo al pedir
    @st.cache_data(show_spinner=False)
    def calc_nonlinear(n: int):
        X = df_full[FEATURES].values
        Xs = StandardScaler().fit_transform(X)
        np.random.seed(42)
        idx = np.random.choice(len(Xs), size=min(n, len(Xs)), replace=False)
        Xs2 = Xs[idx]; y2 = df_full["canceled"].values[idx]
        # t-SNE
        Xt = TSNE(n_components=2, perplexity=30, init="pca",
                  learning_rate="auto", random_state=42, n_jobs=-1).fit_transform(Xs2)
        # LLE
        Xl = LocallyLinearEmbedding(n_components=2, n_neighbors=12,
                                     random_state=42, n_jobs=-1).fit_transform(Xs2)
        # Sammon (fixed 400 pts para velocidad)
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

    # ── PCA siempre visible
    pa, pb = st.columns(2)
    with pa:
        sh("PCA — Scree Plot (varianza explicada)")
        cv = np.cumsum(ev_full)*100
        fig = go.Figure()
        fig.add_trace(go.Bar(x=list(range(1,len(ev_full)+1)), y=ev_full*100,
                              name="Por componente", marker_color=C_OK,
                              marker_line_width=0, opacity=.85))
        fig.add_trace(go.Scatter(x=list(range(1,len(ev_full)+1)), y=cv,
                                  name="Acumulada", mode="lines+markers",
                                  line=dict(color=C_CANCEL,width=2),
                                  marker=dict(size=5)))
        fig.add_hline(y=80, line_dash="dash", line_color="#8B949E",
                      annotation_text="80%", annotation_font_color="#8B949E")
        T(fig,"Varianza por componente")
        fig.update_layout(xaxis_title="Componente", yaxis_title="%", yaxis_range=[0,107])
        st.plotly_chart(fig, use_container_width=True)

    with pb:
        sh("PCA — Loadings PC1 vs PC2")
        ld = pd.DataFrame(comps.T, index=FEATURES, columns=["PC1","PC2"])
        ld = ld.reindex(ld["PC1"].abs().sort_values(ascending=False).index)
        fig = go.Figure()
        fig.add_trace(go.Bar(x=ld.index, y=ld["PC1"], name="PC1",
                              marker_color=C_OK, opacity=.87, marker_line_width=0))
        fig.add_trace(go.Bar(x=ld.index, y=ld["PC2"], name="PC2",
                              marker_color=C_G4, opacity=.87, marker_line_width=0))
        T(fig,"Contribución a PC1 y PC2")
        fig.update_layout(barmode="group", xaxis_tickangle=-38,
                           xaxis_title="", yaxis_title="Loading")
        st.plotly_chart(fig, use_container_width=True)

    sh("PCA — Biplot 2D (1,000 puntos)")
    df_pca_plot = pd.DataFrame({"x":X_pca[:,0],"y":X_pca[:,1],
        "clase":["Cancelada" if v else "No cancelada" for v in y_pca]})
    fig = px.scatter(df_pca_plot, x="x", y="y", color="clase",
                     color_discrete_map={"No cancelada":C_OK,"Cancelada":C_CANCEL},
                     opacity=.45, labels={"x":f"PC1 ({ev2[0]*100:.1f}% var)",
                                          "y":f"PC2 ({ev2[1]*100:.1f}% var)"})
    sc = 3.5
    for i, feat in enumerate(FEATURES):
        fig.add_annotation(x=comps[0,i]*sc, y=comps[1,i]*sc, ax=0, ay=0,
            arrowhead=3, arrowwidth=1.2, arrowcolor="#8B949E",
            text=feat, font=dict(size=8,color="#E6EDF3"),
            bgcolor="rgba(22,27,34,.75)")
    fig.update_traces(marker_size=4)
    T(fig,"PCA Biplot — variables y observaciones")
    st.plotly_chart(fig, use_container_width=True)

    # ── Proyecciones no lineales: botón para calcular
    st.markdown("<div style='height:.5rem'/>", unsafe_allow_html=True)
    sh("Proyecciones no lineales — t-SNE · LLE · Sammon")
    st.info(f"⏱ Calcular con **{proj_n}** puntos (~15-40 s). El resultado se cachea automáticamente.")

    if st.button("▶ Calcular t-SNE, LLE y Mapa de Sammon", type="primary"):
        st.session_state["run_proj"] = True

    if st.session_state.get("run_proj"):
        with st.spinner("Calculando proyecciones no lineales…"):
            Xt, Xl, Xsm, ys, ys_sm = calc_nonlinear(proj_n)

        def proj_fig(X2d, yy, title):
            dp = pd.DataFrame({"x":X2d[:,0],"y":X2d[:,1],
                "clase":["Cancelada" if v else "No cancelada" for v in yy]})
            f = px.scatter(dp, x="x", y="y", color="clase",
                           color_discrete_map={"No cancelada":C_OK,"Cancelada":C_CANCEL},
                           opacity=.45)
            f.update_traces(marker_size=3)
            T(f, title); f.update_layout(height=370, xaxis_title="Dim 1", yaxis_title="Dim 2")
            return f

        pc1, pc2, pc3 = st.columns(3)
        with pc1: st.plotly_chart(proj_fig(Xt, ys,  "t-SNE"), use_container_width=True)
        with pc2: st.plotly_chart(proj_fig(Xl, ys,  "LLE"),   use_container_width=True)
        with pc3: st.plotly_chart(proj_fig(Xsm,ys_sm,"Sammon (n=400)"), use_container_width=True)

        st.success("✅ Proyecciones completadas. PCA (lineal) muestra varianza global. t-SNE y LLE preservan estructura local. Sammon minimiza el estrés de distancias.")

# ──────────────────────────────────────────────────────────────────────────────
# TAB 4 — CORRELACIONES
# ──────────────────────────────────────────────────────────────────────────────
with t4:
    NUM_VARS = ["lead_time","avg_price_per_room","no_of_special_requests",
                "no_of_week_nights","no_of_weekend_nights","no_of_adults",
                "no_of_previous_cancellations","no_of_previous_bookings_not_canceled",
                "total_nights","total_guests","canceled"]

    m_col, s_col = st.columns(2)

    with m_col:
        sh("Método de correlación")
        corr_method = st.radio("",["pearson","spearman","kendall"], horizontal=True,
            format_func=lambda m:{"pearson":"Pearson r","spearman":"Spearman ρ","kendall":"Kendall τ"}[m],
            key="corr_radio")
        cm = df[NUM_VARS].corr(method=corr_method)
        # triangulo inferior
        mask = np.triu(np.ones_like(cm, dtype=bool), k=1)
        cd = cm.where(~mask)
        fig = px.imshow(cd, text_auto=".2f", color_continuous_scale="RdBu_r",
                        zmin=-1, zmax=1, aspect="auto", color_continuous_midpoint=0,
                        labels=dict(color="r/ρ/τ"))
        T(fig, f"Matriz de correlación — {corr_method.capitalize()}")
        fig.update_traces(textfont_size=9)
        st.plotly_chart(fig, use_container_width=True)

    with s_col:
        sh("Coeficientes vs 'canceled'")
        res_list = []
        for col in NUM_VARS:
            if col == "canceled": continue
            rp,pp = pearsonr(df[col], df["canceled"])
            rs,ps = spearmanr(df[col], df["canceled"])
            rk,pk = kendalltau(df[col], df["canceled"])
            res_list.append({"Variable":col,
                "Pearson r":round(rp,4),"p_p":round(pp,5),
                "Spearman ρ":round(rs,4),"p_s":round(ps,5),
                "Kendall τ":round(rk,4),"p_k":round(pk,5)})
        df_cr = pd.DataFrame(res_list).sort_values("Pearson r", key=abs, ascending=False)

        fig = go.Figure()
        xv = df_cr["Variable"]
        for col_n, color, name in [("Pearson r",C_OK,"Pearson r"),
                                    ("Spearman ρ",C_G4,"Spearman ρ"),
                                    ("Kendall τ",C_G3,"Kendall τ")]:
            fig.add_trace(go.Bar(name=name, x=xv, y=df_cr[col_n],
                                  marker_color=color, opacity=.87, marker_line_width=0))
        fig.add_hline(y=0, line_color="#30363D")
        T(fig,"Correlación con 'canceled'")
        fig.update_layout(barmode="group", xaxis_tickangle=-38,
                           yaxis_title="Coeficiente", xaxis_title="")
        st.plotly_chart(fig, use_container_width=True)

    sh("Tabla de coeficientes y significancia")
    df_show = df_cr.copy()
    df_show["Sig."] = df_show["p_p"].apply(lambda p:"✅" if p<0.05 else "❌")
    st.dataframe(df_show.rename(columns={"p_p":"p (P)","p_s":"p (S)","p_k":"p (K)"}),
                 use_container_width=True, hide_index=True)

    sh("Scatter interactivo — variable seleccionada vs precio")
    sel_v = st.selectbox("Variable X:", [v for v in NUM_VARS if v!="canceled"],
                          key="scatter_var")
    fig = px.scatter(df.sample(min(2000,len(df)), random_state=1),
                     x=sel_v, y="avg_price_per_room", color="booking_status",
                     color_discrete_map={"Not_Canceled":C_OK,"Canceled":C_CANCEL},
                     opacity=.5, marginal_x="histogram", marginal_y="histogram",
                     labels={"booking_status":"Estado"})
    fig.update_traces(selector=dict(type="scatter"), marker_size=4)
    T(fig, f"{sel_v} vs Precio — por estado")
    st.plotly_chart(fig, use_container_width=True)

# ──────────────────────────────────────────────────────────────────────────────
# TAB 5 — CHI² & GRANGER
# ──────────────────────────────────────────────────────────────────────────────
with t5:
    # ── Chi-cuadrado
    @st.cache_data
    def calc_chi2(data_hash):
        cat_vars = ["type_of_meal_plan","required_car_parking_space",
                    "repeated_guest","arrival_season","lead_time_cat","total_guests"]
        rows = []
        for var in cat_vars:
            ct = pd.crosstab(df[var], df["booking_status"])
            if ct.shape[0]<2 or ct.shape[1]<2: continue
            chi2, p, dof, _ = chi2_contingency(ct)
            n = ct.sum().sum()
            V = np.sqrt(chi2/(n*(min(ct.shape)-1)))
            rows.append({"Variable":str(var),"χ²":round(chi2,2),"p-valor":round(p,6),
                          "GL":dof,"Cramér's V":round(V,4),"Sig":"✅" if p<0.05 else "❌"})
        return pd.DataFrame(rows).sort_values("Cramér's V",ascending=False)

    df_chi = calc_chi2(len(df))  # hash por tamaño (simple, suficiente)

    cc1, cc2 = st.columns(2)

    with cc1:
        sh("Chi² — Cramér's V por variable")
        fig = px.bar(df_chi, x="Cramér's V", y="Variable", orientation="h",
                     color="Cramér's V",
                     color_continuous_scale=[[0,C_G3],[.35,C_G5],[1,C_CANCEL]],
                     text=df_chi["Cramér's V"].apply(lambda v:f"{v:.3f}"),
                     labels={"Cramér's V":"Cramér's V"})
        fig.update_traces(textposition="outside", marker_line_width=0)
        fig.update_coloraxes(showscale=False)
        T(fig,"Asociación con booking_status"); fig.update_layout(yaxis_title="")
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(df_chi, use_container_width=True, hide_index=True)

    with cc2:
        sh("Causalidad de Granger")
        # series temporales
        @st.cache_data
        def granger_ts():
            d2 = df_full.copy()
            d2["date"] = pd.to_datetime(
                d2[["arrival_year","arrival_month","arrival_date"]]
                .rename(columns={"arrival_year":"year","arrival_month":"month","arrival_date":"day"}),
                errors="coerce")
            mts = d2.groupby(d2["date"].dt.to_period("M")).agg(
                cancel_rate=("canceled","mean"), avg_lead=("lead_time","mean"),
                avg_price=("avg_price_per_room","mean"),
                avg_req=("no_of_special_requests","mean")).reset_index()
            mts["date"] = mts["date"].astype(str)
            return mts.sort_values("date").reset_index(drop=True)

        mts = granger_ts()

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=mts["date"], y=mts["cancel_rate"]*100,
            name="Tasa cancel. (%)", mode="lines+markers",
            line=dict(color=C_CANCEL,width=2), marker=dict(size=6)))
        norm = (mts["avg_lead"]-mts["avg_lead"].mean())/mts["avg_lead"].std()
        fig.add_trace(go.Scatter(x=mts["date"],
            y=norm*5 + mts["cancel_rate"].mean()*100,
            name="Lead time (norm.)", mode="lines+markers",
            line=dict(color=C_OK,width=2,dash="dot"), marker=dict(size=4)))
        T(fig,"Series temporales mensuales")
        fig.update_layout(xaxis_title="Mes", yaxis_title="% / norm.",
                           xaxis_tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)

        # Granger test
        gr_rows = []
        for pred, label in [("avg_lead","Lead time"),("avg_price","Precio"),("avg_req","Solicitudes")]:
            data_g = mts[["cancel_rate",pred]].dropna().values
            if len(data_g) < 7: continue
            res = grangercausalitytests(data_g, maxlag=3, verbose=False)
            for lag in range(1,4):
                f = res[lag][0]["ssr_ftest"][0]; p = res[lag][0]["ssr_ftest"][1]
                gr_rows.append({"Variable":label,"Lag":lag,
                                 "F":round(f,3),"p-valor":round(p,4),
                                 "Causalidad":"✅" if p<0.05 else "—"})
        if gr_rows:
            df_gr = pd.DataFrame(gr_rows)
            pv = df_gr.pivot(index="Variable", columns="Lag", values="p-valor")
            fig = px.imshow(pv, text_auto=".3f",
                            color_continuous_scale=[[0,C_G3],[.05,C_G5],[.2,C_CANCEL],[1,"#3D1A1A"]],
                            zmin=0, zmax=.2, aspect="auto",
                            labels=dict(x="Lag",color="p-valor"))
            T(fig,"Granger — p-valores (verde<0.05 = causalidad)")
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(df_gr, use_container_width=True, hide_index=True)

    sh("Tablas de contingencia — Top 3 variables (Chi²)")
    top3 = df_chi.head(3)["Variable"].tolist()
    tc1, tc2, tc3 = st.columns(3)
    for col_ct, var in zip([tc1,tc2,tc3], top3):
        ctn = pd.crosstab(df[var], df["booking_status"], normalize="index")*100
        ctn_r = ctn.reset_index()
        fig = px.bar(ctn_r, x=var, y=["Canceled","Not_Canceled"],
                     color_discrete_map={"Canceled":C_CANCEL,"Not_Canceled":C_OK},
                     barmode="stack", labels={"value":"% grupo","variable":"Estado"})
        fig.update_traces(marker_line_width=0)
        T(fig, str(var))
        fig.update_layout(height=310, yaxis_title="%", xaxis_title="",
                           xaxis_tickangle=-20)
        col_ct.plotly_chart(fig, use_container_width=True)

# ──────────────────────────────────────────────────────────────────────────────
# TAB 6 — EXPLORADOR
# ──────────────────────────────────────────────────────────────────────────────
with t6:
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
                             color_discrete_sequence=PAL, opacity=.55,
                             hover_data=["booking_status","lead_time","avg_price_per_room"])
    elif chart_type == "Box":
        fig_ex = px.box(df_ex, x=color_var, y=y_var, color=color_var,
                         color_discrete_sequence=PAL, points="outliers")
    elif chart_type == "Violin":
        fig_ex = px.violin(df_ex, x=color_var, y=y_var, color=color_var,
                            color_discrete_sequence=PAL, box=True)
    else:
        fig_ex = px.histogram(df_ex, x=x_var, color=color_var,
                               color_discrete_sequence=PAL,
                               barmode="overlay", opacity=.72, nbins=40)

    fig_ex.update_traces(marker_line_width=0)
    T(fig_ex, f"{x_var} · {y_var} — por {color_var}")
    st.plotly_chart(fig_ex, use_container_width=True)

    sh("Estadísticas descriptivas por grupo")
    group_var = st.selectbox("Agrupar por", cat_all, key="ex_group")
    desc_df = df.groupby(group_var, observed=True)[num_all].describe().round(2)
    st.dataframe(desc_df, use_container_width=True)

    sh("Descargar datos filtrados")
    csv_bytes = df.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Descargar dataset filtrado (.csv)", data=csv_bytes,
                        file_name="hotel_filtrado.csv", mime="text/csv")

# ── FOOTER ────────────────────────────────────────────────────────────────────
st.markdown("""
<hr style='border-color:#30363D;margin:2rem 0 .8rem'/>
<div style='text-align:center;color:#8B949E;font-size:.72rem;padding-bottom:.8rem'>
  Hotel Bookings Analytics · Analítica y Visualización de Datos · Alejandro López<br>
  PCA · t-SNE · LLE · Mapa de Sammon · Pearson · Spearman · Kendall · Chi² · Granger
</div>
""", unsafe_allow_html=True)
