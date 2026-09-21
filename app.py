import streamlit as st
import pandas as pd
import joblib
import os

# ============================================================
# CONFIGURACIÓN DE LA PÁGINA
# ============================================================

st.set_page_config(
    page_title="Clasificador K-NN - Iris",
    page_icon="🌸",
    layout="centered"
)

# ============================================================
# CARGA DEL MODELO Y DEL ESCALADOR
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

modelo = joblib.load(
    os.path.join(BASE_DIR, "modelo_knn_iris.joblib")
)

scaler = joblib.load(
    os.path.join(BASE_DIR, "scaler_iris.joblib")
)

variables = joblib.load(
    os.path.join(BASE_DIR, "variables_iris.joblib")
)

# ============================================================
# ENCABEZADO
# ============================================================

st.title("🌸 Clasificación de agrupaciones con K-NN")

st.write(
    """
    Esta aplicación utiliza un modelo **K-NN** entrenado para reproducir
    las agrupaciones obtenidas previamente mediante **K-Means** sobre
    el conjunto de datos Iris.

    Ingrese las características morfológicas de una nueva muestra para
    determinar a cuál de los clusters identificados pertenece.
    """
)

st.info(
    "El modelo predice clusters generados por K-Means, "
    "no directamente las especies originales de Iris."
)

st.divider()

# ============================================================
# ENTRADA DE DATOS
# ============================================================

st.subheader("Características de la nueva muestra")

col1, col2 = st.columns(2)

with col1:

    sepal_length = st.number_input(
        "Longitud del sépalo (cm)",
        min_value=0.0,
        max_value=10.0,
        value=5.1,
        step=0.1
    )

    petal_length = st.number_input(
        "Longitud del pétalo (cm)",
        min_value=0.0,
        max_value=10.0,
        value=1.4,
        step=0.1
    )

with col2:

    sepal_width = st.number_input(
        "Ancho del sépalo (cm)",
        min_value=0.0,
        max_value=10.0,
        value=3.5,
        step=0.1
    )

    petal_width = st.number_input(
        "Ancho del pétalo (cm)",
        min_value=0.0,
        max_value=10.0,
        value=0.2,
        step=0.1
    )

# ============================================================
# CREACIÓN DE LA OBSERVACIÓN
# ============================================================

nueva_muestra = pd.DataFrame(
    [[
        sepal_length,
        sepal_width,
        petal_length,
        petal_width
    ]],
    columns=variables
)

st.divider()

# ============================================================
# MOSTRAR DATOS INGRESADOS
# ============================================================

with st.expander("Ver datos ingresados"):

    muestra_visual = nueva_muestra.copy()

    muestra_visual.columns = [
        "Longitud sépalo",
        "Ancho sépalo",
        "Longitud pétalo",
        "Ancho pétalo"
    ]

    st.dataframe(
        muestra_visual,
        use_container_width=True
    )

# ============================================================
# PREDICCIÓN
# ============================================================

if st.button(
    "Clasificar muestra",
    type="primary",
    use_container_width=True
):

    # Aplicar el mismo escalamiento utilizado en entrenamiento
    muestra_escalada = scaler.transform(nueva_muestra)

    # Predicción mediante K-NN
    cluster = int(
        modelo.predict(muestra_escalada)[0]
    )

    st.subheader("Resultado de la clasificación")

    if cluster == 0:

        st.success("La nueva muestra pertenece al **Cluster 0**.")

        st.write(
            """
            **Perfil general del segmento:** este cluster corresponde
            al grupo de Iris con mayores dimensiones de pétalo y sépalo
            en comparación con el segundo segmento identificado por
            K-Means.
            """
        )

    elif cluster == 1:

        st.success("La nueva muestra pertenece al **Cluster 1**.")

        st.write(
            """
            **Perfil general del segmento:** este cluster presenta
            pétalos considerablemente más pequeños y sépalos
            relativamente más anchos. En el análisis exploratorio,
            este segmento coincidió con las observaciones de Iris
            setosa.
            """
        )

    else:

        st.warning(
            f"El modelo asignó la observación al Cluster {cluster}."
        )

    # Distancias a los vecinos más cercanos
    distancias, indices = modelo.kneighbors(
        muestra_escalada
    )

    st.metric(
        "Distancia al vecino más cercano",
        f"{distancias[0][0]:.4f}"
    )

# ============================================================
# INFORMACIÓN METODOLÓGICA
# ============================================================

st.divider()

with st.expander("¿Cómo funciona el modelo?"):

    st.write(
        """
        1. La aplicación recibe las cuatro características de la muestra.
        2. Los valores son transformados utilizando el mismo
           **StandardScaler** empleado durante el entrenamiento.
        3. El modelo **K-NN**, configurado con K = 1, busca la observación
           de entrenamiento más cercana.
        4. La nueva muestra recibe la etiqueta del cluster correspondiente
           a su vecino más cercano.
        5. Las etiquetas utilizadas fueron generadas previamente mediante
           K-Means, por lo que representan segmentos no supervisados.
        """
    )

st.caption(
    "Taller 6 de Inteligencia Artificial | "
    "K-Means + K-NN + despliegue interactivo"
)
