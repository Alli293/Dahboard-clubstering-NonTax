# ============================================================
# 🚀 STREAMLIT DASHBOARD – CLUSTERING SEMÁNTICO (Versión modificada)
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import re

# ---------------------------
# CONFIG
# ---------------------------
st.set_page_config(page_title="Dashboard – Clustering Semántico", layout="wide")
st.title("Dashboard – Clustering Semántico (versión personalizada)")

# ---------------------------
# CONSTANTES
# ---------------------------
COL_TITULO = "puesto_cluster_ready"
COL_CLUSTER = "cluster_refinado_sub"
COL_CAT_ORIGINAL = "Categoría"
COL_SILHOUETTE = "silhouette_score"

# === USAR ESTE DICCIONARIO (SIN EMOJIS PARA WORDCLOUD) ===
CLUSTER_MAP = {
    "0": "Ingenieros de Software",
    "1": "Especialistas en electromecánico/ingeniero",
    "3": "Especialistas en security/analyst",
    "4": "Especialistas en producción/operario",
    "5": "Representantes de Ventas o Marketing",
    "6": "Especialistas en engineer/quality",
    "8": "Analistas o Científicos de Datos",
    "9": "Profesores o Instructores de Idiomas",
    "10": "Especialistas en coordinador/planner",
    "12": "Especialistas en operario/calificado",
    "13": "Especialistas en asesor/técnico",
    "14": "Especialistas en médico/equipo",
    "15": "Personal Administrativo",
    "16": "Especialistas en ejecutivo/encargado"
}

# ---------------------------
# CARGAR CSV
# ---------------------------
@st.cache_data
def load_data(path="dataset_clustering_semantico_2nivel_nombres.csv"):
    df = pd.read_csv(path)
    return df

df = load_data()

# ---------------------------
# LIMPIEZA DE CATEGORÍAS (MISMA LÓGICA DE ANTES)
# ---------------------------
def clean_categories(df):
    df = df.copy()
    df[COL_CAT_ORIGINAL] = df[COL_CAT_ORIGINAL].astype(str).fillna("").str.strip()
    df[COL_TITULO] = df[COL_TITULO].astype(str).fillna("")

    patrones = r"(administración|oficina|admin|educación|docencia|docente|profesor|enseñanza)"
    mask_orig = df[COL_CAT_ORIGINAL].str.contains(patrones, case=False)

    return df[~mask_orig]

df = clean_categories(df)

# ============================================================
#  ASIGNAR CATEGORÍA SEMÁNTICA SEGÚN TU DICCIONARIO
# ============================================================
df["cluster_base"] = df[COL_CLUSTER].astype(str).str.extract(r"(\d+)")

df["categoria_semantica_final"] = df["cluster_base"].map(CLUSTER_MAP).fillna("Sin categoría")

# ---------------------------
# FILTROS LATERALES
# ---------------------------
st.sidebar.header("Filtros")
min_cluster_size = st.sidebar.slider("Excluir categorías con menos de X registros:", 0, 200, 3)
top_src = st.sidebar.slider("Top puestos para Sankey:", 3, 30, 10)
top_tgt = st.sidebar.slider("Top categorías semánticas:", 3, 16, 10)

# ---------------------------
# FILTRAR CATEGORÍAS PEQUEÑAS
# ---------------------------
valid_cats = df["categoria_semantica_final"].value_counts()
valid_cats = valid_cats[valid_cats >= min_cluster_size].index
df = df[df["categoria_semantica_final"].isin(valid_cats)]

# ---------------------------
# MÉTRICAS
# ---------------------------
st.subheader(" Métricas Generales")
c1, c2, c3 = st.columns(3)
c1.metric("Total registros", len(df))
c2.metric("Puestos únicos", df[COL_TITULO].nunique())
c3.metric("Categorías semánticas", df["categoria_semantica_final"].nunique())

st.markdown("---")

# ============================================================
#  SANKEY USANDO PUESTOS → CATEGORÍA SEMÁNTICA
# ============================================================
def prepare_sankey(df, top_src, top_tgt):
    top_puestos = df[COL_TITULO].value_counts().nlargest(top_src).index.tolist()
    top_cats = df["categoria_semantica_final"].value_counts().nlargest(top_tgt).index.tolist()

    df_f = df[df[COL_TITULO].isin(top_puestos) & df["categoria_semantica_final"].isin(top_cats)]

    agg = df_f.groupby([COL_TITULO, "categoria_semantica_final"]).size().reset_index(name="count")

    nodes_src = list(agg[COL_TITULO].unique())
    nodes_tgt = list(agg["categoria_semantica_final"].unique())
    nodes = nodes_src + nodes_tgt

    node_index = {label: i for i, label in enumerate(nodes)}

    sources = agg[COL_TITULO].map(node_index).tolist()
    targets = agg["categoria_semantica_final"].map(node_index).tolist()
    values = agg["count"].tolist()

    return nodes, sources, targets, values

# ---------------------------
# SANKEY FINAL
# ---------------------------
st.subheader(" Sankey: Puesto → Categoría Semántica")

nodes, sources, targets, values = prepare_sankey(df, top_src, top_tgt)

if len(values) == 0:
    st.warning("No hay datos suficientes para construir el Sankey.")
else:
    fig_sankey = go.Figure(data=[go.Sankey(
        arrangement="snap",
        node=dict(
            pad=12,
            thickness=16,
            line=dict(color="black", width=0.2),
            label=nodes,
            color="#666"
        ),
        link=dict(
            source=sources,
            target=targets,
            value=values,
            color="rgba(0,0,0,0.25)"
        )
    )])
    fig_sankey.update_layout(height=750)
    st.plotly_chart(fig_sankey, use_container_width=True)

st.markdown("---")

# ============================================================
#  WORDCLOUD (usando TU categoría semántica, sin emojis)
# ============================================================
st.subheader(" Nube de Palabras")

options_sem = sorted(df["categoria_semantica_final"].unique())
cat_sel = st.selectbox("Selecciona categoría:", options_sem)

text = " ".join(df[df["categoria_semantica_final"] == cat_sel][COL_TITULO])

wc = WordCloud(width=1200, height=450, background_color="white").generate(text)
fig_wc, ax = plt.subplots(figsize=(14, 5))
ax.imshow(wc)
ax.axis("off")
st.pyplot(fig_wc)
