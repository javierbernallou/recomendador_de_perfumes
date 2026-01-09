import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import json
from tensorflow.keras.models import load_model

st.set_page_config(page_title="Perfume AI Recommender", layout="wide")

FEATURES = ["mainaccord1", "mainaccord2", "mainaccord3", "mainaccord4", "mainaccord5"]

@st.cache_resource
def load_assets():
    # 1) Data
    df = pd.read_csv("Libro3.csv", sep=";", encoding="latin1")

    # 2) Mapping (categorical accord -> number)
    with open("mapping.json", "r", encoding="utf-8") as f:
        mapping = json.load(f)

    X = df[FEATURES].fillna("none")
    X_num = X.applymap(lambda x: mapping.get(str(x), 0)).astype(float)

    # 3) Scaler + PCA entrenados en notebook
    scaler = joblib.load("scaler.joblib")
    pca = joblib.load("pca.joblib")

    X_scaled = scaler.transform(X_num)
    pcs = pca.transform(X_scaled)
    pcadf = pd.DataFrame(pcs, columns=["PC1", "PC2", "PC3", "PC4"], index=df.index)

    # 4) Model
    model = load_model("modelo_perfumes2.keras", compile=False)

    return df, pcadf, model

def cosine_sim_matrix(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    # A: (n,4), b: (4,) or (1,4)
    b = b.reshape(1, -1)
    An = np.linalg.norm(A, axis=1, keepdims=True) + 1e-9
    bn = np.linalg.norm(b, axis=1, keepdims=True) + 1e-9
    return (A @ b.T) / (An * bn)  # (n,1)

def euclidean_dist(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    b = b.reshape(1, -1)
    return np.linalg.norm(A - b, axis=1)  # (n,)

def plot_market(
    pcadf: pd.DataFrame,
    top10_nn_idx,
    top10_cos_idx,
    perfume_ref: str,
    ref_idx,
    xcol: str,
    ycol: str,
    title: str
):
    fig, ax = plt.subplots(figsize=(7, 5))

    # Fondo: todos los perfumes
    sns.scatterplot(
        data=pcadf, x=xcol, y=ycol,
        color="lightgrey", alpha=0.25, s=12, ax=ax
    )

    # Top-10 Coseno (sin red) -> VERDE
    if top10_cos_idx is not None and len(top10_cos_idx) > 0:
        cos_pcs = pcadf.loc[top10_cos_idx]
        sns.scatterplot(
            x=cos_pcs[xcol], y=cos_pcs[ycol],
            color="green", s=90, ax=ax, label="Top-10 Coseno"
        )

    # Top-10 NN -> ROJO
    if top10_nn_idx is not None and len(top10_nn_idx) > 0:
        nn_pcs = pcadf.loc[top10_nn_idx]
        sns.scatterplot(
            x=nn_pcs[xcol], y=nn_pcs[ycol],
            color="red", s=90, ax=ax, label="Top-10 NN"
        )

    # Referencia -> AZUL
    if perfume_ref != "Ninguno" and ref_idx is not None:
        refp = pcadf.loc[ref_idx]
        ax.scatter(
            refp[xcol], refp[ycol],
            color="blue", s=180, edgecolors="black", label="Referencia"
        )

    ax.set_title(title)
    ax.legend()
    st.pyplot(fig)

# ---- Load assets
try:
    df, pcadf, model = load_assets()
except Exception as e:
    st.error(f"Error al cargar activos: {e}")
    st.stop()

# ---- Sidebar
st.sidebar.header("Panel de Control")

perfume_ref = st.sidebar.selectbox(
    "Perfume de referencia (opcional)",
    ["Ninguno"] + sorted(df["Perfume"].dropna().unique().tolist())
)

# Defaults: si hay referencia, usa sus PCs
default_pc = np.array([0.0, 0.0, 0.0, 0.0], dtype=float)
ref_idx = None
if perfume_ref != "Ninguno":
    ref_idx = df.index[df["Perfume"] == perfume_ref][0]
    default_pc = pcadf.loc[ref_idx, ["PC1", "PC2", "PC3", "PC4"]].values.astype(float)

st.sidebar.subheader("Tu perfil (PCA)")
pc1 = st.sidebar.slider(
    "PC1 (fresco ↔ intenso)",
    float(pcadf["PC1"].min()), float(pcadf["PC1"].max()), float(default_pc[0])
)
pc2 = st.sidebar.slider(
    "PC2 (limpio ↔ dulce)",
    float(pcadf["PC2"].min()), float(pcadf["PC2"].max()), float(default_pc[1])
)
pc3 = st.sidebar.slider(
    "PC3 (herbal ↔ especiado)",
    float(pcadf["PC3"].min()), float(pcadf["PC3"].max()), float(default_pc[2])
)
pc4 = st.sidebar.slider(
    "PC4 (cuero ↔ vainilla)",
    float(pcadf["PC4"].min()), float(pcadf["PC4"].max()), float(default_pc[3])
)

use_exact_pca = st.sidebar.checkbox(
    "Usar PCs exactos del perfume de referencia (ignorar sliders)",
    value=False,
    help="Sirve para testear que Coseno y NN están comparando el mismo espacio."
)

debug = st.sidebar.checkbox("Modo debug", value=True)

# user vector final
user_vector = np.array([pc1, pc2, pc3, pc4], dtype=float)
if perfume_ref != "Ninguno" and use_exact_pca:
    user_vector = pcadf.loc[ref_idx, ["PC1", "PC2", "PC3", "PC4"]].values.astype(float)

# Filtro de género (robusto)
st.sidebar.subheader("Filtro")
gender_values = sorted(df["Gender"].dropna().unique().tolist()) if "Gender" in df.columns else []
generos = st.sidebar.multiselect(
    "Género deseado",
    options=gender_values,
    default=gender_values
)

# ---- Main
st.title("Perfume Neural Matcher")
st.write("Recomendador basado en PCA + Cosine + Red Neuronal.")

if debug:
    st.subheader("Debug: vector usuario")
    st.write(user_vector)

if st.button("Calcular afinidades"):
    # 1) Filtrado por género (si no eliges nada, no filtra)
    if ("Gender" in df.columns) and (generos is not None) and (len(generos) > 0):
        mask = df["Gender"].isin(generos)
    else:
        mask = np.ones(len(df), dtype=bool)

    dff = df.loc[mask].copy()
    pcaf = pcadf.loc[mask].values  # (n,4)

    # 2) Coseno + Distancia (en PCs reales)
    cos_scores = cosine_sim_matrix(pcaf, user_vector).flatten()
    euc_dist = euclidean_dist(pcaf, user_vector)

    dff["Cosine"] = cos_scores
    dff["EucDist"] = euc_dist

    top10_cos = dff.sort_values("Cosine", ascending=False).head(10)
    top10_euc = dff.sort_values("EucDist", ascending=True).head(10)

    # 3) NN (mismo input: (user, perfume))
    user_input = np.tile(user_vector.reshape(1, -1), (len(pcaf), 1))
    scores = model.predict([user_input, pcaf], verbose=0).flatten()  # 0..1 si sigmoid
    dff["Match"] = np.round(scores * 100, 2)

    top10_nn = dff.sort_values("Match", ascending=False).head(10)

    # 4) Debug tables
    if debug:
        st.subheader("Debug: Top-10 por Cosine (sin red)")
        st.dataframe(
            top10_cos[["Perfume", "Brand", "Gender", "Cosine"]].reset_index(drop=True),
            use_container_width=True
        )

        st.subheader("Debug: Top-10 por Euclidea (sin red)")
        st.dataframe(
            top10_euc[["Perfume", "Brand", "Gender", "EucDist"]].reset_index(drop=True),
            use_container_width=True
        )

        st.subheader("Debug: Top-10 por Red Neuronal")
        st.dataframe(
            top10_nn[["Perfume", "Brand", "Gender", "Match"]].reset_index(drop=True),
            use_container_width=True
        )

        overlap = len(set(top10_nn.index).intersection(set(top10_cos.index)))
        st.write(f"Coincidencias NN vs Cosine (top10): {overlap}/10")

    # 5) Resultados finales
    st.subheader("Tus mejores coincidencias (Red Neuronal)")
    st.dataframe(
        top10_nn[["Perfume", "Brand", "Gender", "Match"]],
        use_container_width=True,
        hide_index=True
    )

    # 6) Plots
    st.divider()

    # Radar + 2D plots
    col1, col2 = st.columns(2)

    with col1:
        st.write("Perfil (PC1..PC4)")
        labels = ["PC1", "PC2", "PC3", "PC4"]
        angles = np.linspace(0, 2*np.pi, len(labels), endpoint=False).tolist()
        angles += angles[:1]
        stats = user_vector.tolist() + [user_vector[0]]

        fig, ax = plt.subplots(figsize=(5, 5), subplot_kw=dict(polar=True))
        ax.fill(angles, stats, alpha=0.25)
        ax.plot(angles, stats, linewidth=2)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(labels)
        st.pyplot(fig)

    with col2:
        st.write("Mercado (PC1 vs PC2) — Verde=Coseno, Rojo=NN")
        plot_market(
            pcadf=pcadf,
            top10_nn_idx=top10_nn.index,
            top10_cos_idx=top10_cos.index,
            perfume_ref=perfume_ref,
            ref_idx=ref_idx,
            xcol="PC1",
            ycol="PC2",
            title="Mercado (PC1 vs PC2)"
        )

    # Segundo gráfico debajo (no dentro de col2)
    st.write("Mercado (PC3 vs PC4) — Verde=Coseno, Rojo=NN")
    plot_market(
        pcadf=pcadf,
        top10_nn_idx=top10_nn.index,
        top10_cos_idx=top10_cos.index,
        perfume_ref=perfume_ref,
        ref_idx=ref_idx,
        xcol="PC3",
        ycol="PC4",
        title="Mercado (PC3 vs PC4)"
    )
