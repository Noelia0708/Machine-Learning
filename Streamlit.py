import streamlit as st
import pandas as pd
import joblib

# ======================================================
# 🎨 CONFIGURACIÓN DE LA PÁGINA (SIEMPRE PRIMERO)
# ======================================================
st.set_page_config(
    page_title="SafeTrip USA",
    page_icon="🛡️",
    layout="wide"
)

# ======================================================
# 🔧 CARGAR MODELO, SCALER Y FEATURES
# ======================================================
@st.cache_resource
def load_artifacts():
    model = joblib.load("regressionlineal_model.pkl")
    scaler = joblib.load("scaler.pkl")

    with open("selected_features.txt", "r") as f:
        feature_names = [line.strip() for line in f.readlines()]

    return model, scaler, feature_names

model, scaler, feature_names = load_artifacts()

# ❗ ELIMINAMOS EL TARGET SI VINIERA EN EL TXT
if "ViolentCrimesPerPop" in feature_names:
    feature_names.remove("ViolentCrimesPerPop")

# ======================================================
# 📌 DICCIONARIO DE NOMBRES EN ESPAÑOL 
# ======================================================
pretty_names = {
    "PctIlleg": "Niños nacidos fuera del matrimonio",
    "racepctblack": "Población negra (%)",
    "pctWPubAsst": "Asistencia pública (población blanca)",
    "FemalePctDiv": "Mujeres divorciadas (%)",
    "TotalPctDiv": "Personas divorciadas (%)",
    "MalePctDivorce": "Hombres divorciados (%)",
    "PctPopUnderPov": "Población bajo el umbral de pobreza (%)",
    "PctUnemployed": "Desempleo (%)",
    "PctHousNoPhone": "Viviendas sin teléfono (%)",
    "PctNotHSGrad": "Personas sin educación secundaria (%)",
    "PctVacantBoarded": "Viviendas vacías y tapiadas (%)",
    "PctHousLess3BR": "Viviendas con menos de 3 habitaciones (%)",
    "NumIlleg": "Numero de actividades ilegales",
    "PctPersOwnOccup": "Personas en vivienda propia (%)",
    "pctWInvInc": "Ingresos por inversión (población blanca)",
    "PctTeen2Par": "Adolescentes con dos padres (%)",
    "PctYoungKids2Par": "Niños pequeños con dos padres (%)",
    "racePctWhite": "Población blanca (%)",
    "PctFam2Par": "Familias con dos padres (%)",
    "PctKids2Par": "Niños con dos padres (%)"
}

# ======================================================
# 🧭 NAVEGACIÓN
# ======================================================
st.sidebar.title("🧭 Navegación")
pagina = st.sidebar.radio("Ir a:", ["🏠 Inicio", "🧪 Test de Peligrosidad", "📤 Subir Archivo", "🚀 Próximos Pasos"])


# ======================================================
# 🏠 PÁGINA 1 — INICIO
# ======================================================
if pagina == "🏠 Inicio":
    st.title("🛡️ SafeTrip USA — Analiza la peligrosidad antes de viajar")

    st.markdown("""
    ### ✈️ Tu compañera de seguridad para viajes por Estados Unidos  
    SafeTrip USA evalúa el nivel de peligrosidad de un área basándose en factores sociales,
    económicos, demográficos y policiales.

    ### 🔍 Índice de criminalidad (0–1)
    - **0 – 0.33 → Zona Segura 🟢**
    - **0.34 – 0.66 → Zona con Riesgo Medio 🟡**
    - **0.67 – 1.00 → Zona de Alto Riesgo 🔴**

    ### 🧲 Eslogan oficial:
    ## *"Viaja tranquilo. Viaja seguro. SafeTrip USA te acompaña."*
    """)


# ======================================================
# 🧪 PÁGINA 2 — TEST INTERACTIVO
# ======================================================
elif pagina == "🧪 Test de Peligrosidad":
    st.title("🧪 Test — Calcula la peligrosidad de un área")

    st.write("Selecciona el nivel de cada variable:")

    nivel_a_valor = {
        "Bajo": 0.0,
        "Medio": 0.5,
        "Alto": 1.0
    }

    opciones = ["Bajo", "Medio", "Alto"]

    input_values = {}

    # Inputs en el orden original
    for feature in feature_names:
        label = pretty_names.get(feature, feature)
        opcion = st.selectbox(label, opciones, index=1)
        input_values[feature] = nivel_a_valor[opcion]

    if st.button("🔮 Calcular peligrosidad"):
        df_input = pd.DataFrame([input_values])
        df_input = df_input[feature_names]

        scaled = scaler.transform(df_input)
        pred = model.predict(scaled)[0]

        st.success(f"🔎 Índice estimado de criminalidad: **{round(pred, 4)}**")

        if pred <= 0.33:
            st.success("🟢 Zona Segura")
        elif pred <= 0.66:
            st.warning("🟡 Zona con Riesgo Medio")
        else:
            st.error("🔴 Zona de Alto Riesgo")


# ======================================================
# 📤 PÁGINA 3 — SUBIR ARCHIVO CON MAPEO DE COLUMNAS
# ======================================================
elif pagina == "📤 Subir Archivo":
    st.title("📤 Subir Archivo CSV para clasificar varias zonas")

    file = st.file_uploader("Sube un archivo CSV", type=["csv"])

    if file:
        df = pd.read_csv(file)
        st.write("Columnas detectadas en tu archivo:")
        st.write(list(df.columns))

        st.subheader("🧩 Mapeo de columnas")
        st.markdown("""
        Selecciona qué columna de tu archivo corresponde a cada variable del modelo.
        Si alguna variable no existe, déjala en **'---'** (se rellenará con 0).
        """)

        column_map = {}
        columnas_usuario = ["---"] + list(df.columns)

        for feature in feature_names:
            pretty = pretty_names.get(feature, feature)
            seleccion = st.selectbox(
                f"{pretty} → ({feature})",
                columnas_usuario,
                index=0
            )
            column_map[feature] = seleccion

        if st.button("🔄 Aplicar mapeo y calcular"):
            if all(column_map[f] == "---" for f in feature_names):
                st.error("❌ No se ha mapeado ninguna columna.")
            else:
                df_aligned = pd.DataFrame()

                for feature in feature_names:
                    col = column_map[feature]
                    if col == "---":
                        df_aligned[feature] = 0
                    else:
                        df_aligned[feature] = df[col]

                scaled = scaler.transform(df_aligned)
                df_aligned["predicted_risk"] = model.predict(scaled)

                st.success("✔️ Archivo procesado correctamente")
                st.dataframe(df_aligned)

                st.download_button(
                    "⬇️ Descargar resultados",
                    df_aligned.to_csv(index=False).encode("utf-8"),
                    file_name="resultados_mapeados.csv",
                    mime="text/csv"
                )


# ======================================================
# 🚀 PÁGINA 4 — PRÓXIMOS PASOS
# ======================================================
elif pagina == "🚀 Próximos Pasos":
    st.title("🚀 Próximos pasos del proyecto SafeTrip USA")

    st.markdown("""
    ### 🔮 Mejoras futuras:
    - Más datos policiales detallados  
    - Variables socioeconómicas adicionales  
    - Modelos por estado  
    - Mapas interactivos por riesgo  
    - App móvil con GPS  

    ### 🎯 Objetivo
    Detectar y avisarte en tiempo real cuando entres en una zona peligrosa.

    Gracias por usar SafeTrip USA.
    """)
