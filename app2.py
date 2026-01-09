import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from tensorflow.keras.models import load_model
from tensorflow.keras.losses import MeanSquaredError

# Configuración visual de la página
st.set_page_config(page_title="Perfume AI Recommender", layout="wide")

# 1. CARGA DE DATOS Y MODELO (Caché para optimizar)
@st.cache_resource
def load_assets():
    # Carga de datos
    df = pd.read_csv("Libro2.csv", sep=';', encoding='latin1')
    
    # Procesamiento PCA (debe ser idéntico a tu Jupyter)
    features = ['mainaccord1', 'mainaccord2', 'mainaccord3', 'mainaccord4', 'mainaccord5']
    X = df[features].fillna('none')
    all_accords = sorted(pd.unique(X.values.ravel()))
    mapping = {accord: i for i, accord in enumerate(all_accords)}
    X_numeric = X.applymap(lambda x: mapping.get(x))
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_numeric)
    
    pca = PCA(n_components=4, random_state=0)
    pcs = pca.fit_transform(X_scaled)
    pca_df = pd.DataFrame(pcs, columns=['PC1', 'PC2', 'PC3', 'PC4'], index=df.index)
    
    # Cargar la Red Neuronal entrenada
    # Asegúrate de haber hecho model.save('modelo_perfumes.h5') en Jupyter
    model = load_model('modelo_perfumes.h5', custom_objects={'mse': MeanSquaredError()})        
    return df, pca_df, model

try:
    df, pca_df, model = load_assets()
except Exception as e:
    st.error(f"Error al cargar activos: {e}")
    st.stop()

# --- INTERFAZ LATERAL (SIDEBAR) ---
st.sidebar.header("🎛️ Panel de Control Olfativo")

# BUSCADOR DE REFERENCIA (Opcional)
st.sidebar.subheader("¿Ya tienes un perfume favorito?")
perfume_ref = st.sidebar.selectbox("Selecciona para copiar su perfil:", ["Ninguno"] + list(df['Perfume'].unique()))

# Valores por defecto de los sliders
default_vals = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
if perfume_ref != "Ninguno":
    idx = df[df['Perfume'] == perfume_ref].index[0]
    # Inversa aproximada de los componentes para los sliders
    row = pca_df.loc[idx]
    default_vals = [row['PC1'], row['PC1'], row['PC2'], row['PC2'], row['PC3'], row['PC4']]

# SLIDERS DESGLOSADOS
st.sidebar.subheader("Ajuste de Notas")
f1 = st.sidebar.slider("Jabón / Limpieza", -3.0, 3.0, float(default_vals[0]))
f2 = st.sidebar.slider("Pino / Amargo", -3.0, 3.0, float(default_vals[1]))
c1 = st.sidebar.slider("Especias / Oriental", -3.0, 3.0, float(default_vals[2]))
c2 = st.sidebar.slider("Dulzura / Vainilla", -3.0, 3.0, float(default_vals[3]))
p1 = st.sidebar.slider("Hierbas / Natural", -3.0, 3.0, float(default_vals[4]))
p2 = st.sidebar.slider("Cuero / Animalic", -3.0, 3.0, float(default_vals[5]))

# Filtro de Género
generos = st.sidebar.multiselect("Género deseado:", df['Gender'].unique(), default=df['Gender'].unique())

# Construcción del vector de entrada (4 componentes)
user_vector = np.array([(f1+f2)/2, (c1+c2)/2, p1, p2])
debug = st.sidebar.checkbox("🧰 Modo debug", value=True)
use_exact_pca = st.sidebar.checkbox("🔁 Debug: usar PCs exactos del perfume (ignorar sliders)", value=False)

if perfume_ref != "Ninguno" and use_exact_pca:
    user_vector = pca_df.loc[idx, ['PC1','PC2','PC3','PC4']].values.astype(float)


if debug:
    st.sidebar.write("### Debug: vectores")
    st.sidebar.write("perfume_ref:", perfume_ref)
    if perfume_ref != "Ninguno":
        st.sidebar.write("idx:", int(idx))
        st.sidebar.write("PCs perfume (PC1..PC4):")
        st.sidebar.dataframe(pca_df.loc[idx, ['PC1','PC2','PC3','PC4']].to_frame().T)
    st.sidebar.write("user_vector (entrada usuario):", user_vector)


# --- CUERPO PRINCIPAL ---
st.title("🧪 Perfume Neural Matcher")
st.write("Nuestra IA analiza 7,499 fragancias para encontrar tu combinación química perfecta.")

def cosine_sim_matrix(A, b):
    # A: (n, d), b: (d,)
    b = b.reshape(1, -1)
    A_norm = np.linalg.norm(A, axis=1, keepdims=True) + 1e-9
    b_norm = np.linalg.norm(b, axis=1, keepdims=True) + 1e-9
    return (A @ b.T) / (A_norm * b_norm)  # (n,1)

def euclidean_dist(A, b):
    return np.linalg.norm(A - b.reshape(1, -1), axis=1)  # (n,)


if st.button('🚀 Calcular Afinidades'):
    # Filtrado por género
    mask = df['Gender'].isin(generos)
    df_f = df[mask].copy()
    pca_f = pca_df[mask].values
    # --- Similitud directa en PCA (sin red) ---
    cos_scores = cosine_sim_matrix(pca_f, user_vector).flatten()
    euc_dist = euclidean_dist(pca_f, user_vector)

    df_f['Cosine'] = cos_scores
    df_f['EucDist'] = euc_dist

    top10_cos = df_f.sort_values('Cosine', ascending=False).head(10)
    top10_euc = df_f.sort_values('EucDist', ascending=True).head(10)

    if debug:
        st.subheader("🧪 Debug: Top-10 por Cosine (sin red)")
        st.dataframe(top10_cos[['Perfume','Brand','Gender','Cosine']].reset_index(drop=True), use_container_width=True)

        st.subheader("🧪 Debug: Top-10 por Distancia Euclídea (sin red)")
        st.dataframe(top10_euc[['Perfume','Brand','Gender','EucDist']].reset_index(drop=True), use_container_width=True)

    
    # Predicción con la Red Neuronal
    user_input = np.tile(user_vector, (len(pca_f), 1))
    scores = model.predict([user_input, pca_f], verbose=0)
    
    # Normalización del Score a 0-100%
    # (Si tu salida es Sigmoide ya está entre 0-1, si no, escalamos)
    match_pct = (scores.flatten() * 100).round(2)
    df_f['Match %'] = match_pct
    
    # Top 10 resultados
    top_10 = df_f.sort_values('Match %', ascending=False).head(10)
    if debug:
        st.subheader("🧪 Debug: Top-10 por Red Neuronal")
        st.dataframe(top_10[['Perfume','Brand','Gender','Match %']].reset_index(drop=True), use_container_width=True)

        # Intersección (cuánto coinciden)
        set_nn = set(top_10.index)
        set_cos = set(top10_cos.index)
        overlap = len(set_nn.intersection(set_cos))
        st.write(f"Coincidencias NN vs Cosine (top10): {overlap}/10")

    
    # Mostrar resultados con barra de progreso
    st.subheader("🎯 Tus Mejores Coincidencias")
    st.dataframe(
        top_10[['Perfume', 'Brand', 'Gender', 'Match %']],
        column_config={
            "Match %": st.column_config.ProgressColumn(
                "Nivel de Afinidad",
                help="Porcentaje de match calculado por la Red Neuronal",
                format="%f%%",
                min_value=0,
                max_value=100,
            ),
        },
        use_container_width=True,
        hide_index=True
    )

    # --- VISUALIZACIONES ---
    st.divider()
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("### 🕸️ Tu Huella Olfativa")
        labels = ['Fresco', 'Cálido', 'Herbal', 'Intenso']
        angles = np.linspace(0, 2*np.pi, len(labels), endpoint=False).tolist()
        angles += angles[:1]
        stats = np.concatenate((user_vector, [user_vector[0]]))
        
        fig, ax = plt.subplots(figsize=(5, 5), subplot_kw=dict(polar=True))
        ax.fill(angles, stats, color='magenta', alpha=0.25)
        ax.plot(angles, stats, color='magenta', linewidth=2)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(labels)
        st.pyplot(fig)

    with col2:
        st.write("### 📍 Posicionamiento en el Mercado")
        fig2, ax2 = plt.subplots(figsize=(7, 5))
        sns.scatterplot(data=pca_df, x='PC1', y='PC2', color='lightgrey', alpha=0.3, ax=ax2)
        
        # Resaltar recomendados
        rec_pcs = pca_df.loc[top_10.index]
        sns.scatterplot(x=rec_pcs['PC1'], y=rec_pcs['PC2'], color='red', s=100, ax=ax2, label="Recomendados")
        
        # Añadir etiquetas a los 3 primeros
        for i in range(3):
            ax2.text(rec_pcs['PC1'].iloc[i]+0.1, rec_pcs['PC2'].iloc[i], top_10['Perfume'].iloc[i], fontsize=8)
            
        plt.legend()
        st.pyplot(fig2)

    if debug:
        st.subheader("📊 Debug: rangos de PC1..PC4")
        desc = pca_df[['PC1','PC2','PC3','PC4']].describe(percentiles=[.01,.05,.25,.5,.75,.95,.99])
        st.dataframe(desc, use_container_width=True)

        st.write("user_vector:", user_vector)

    if debug:
        fig, axs = plt.subplots(2, 2, figsize=(10, 6))
        cols = ['PC1','PC2','PC3','PC4']
        for ax, c in zip(axs.ravel(), cols):
            ax.hist(pca_df[c].values, bins=50, alpha=0.7)
            ax.axvline(user_vector[cols.index(c)], color='red', linewidth=2)
            ax.set_title(c)
        plt.tight_layout()
        st.pyplot(fig)
