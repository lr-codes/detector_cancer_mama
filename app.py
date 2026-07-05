import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib

st.set_page_config(page_title="Detector de Cáncer de Mama", layout="centered")

st.title("Sistema de Diagnóstico de Cáncer de Mama")
st.write("Esta aplicación utiliza un modelo de Machine Learning (Bosque Aleatorio) auditado con una precisión del **95.6%** para predecir si un tumor celular es benigno o maligno.")

@st.cache_resource
def cargar_modelo():
    return joblib.load('modelo_cancer.joblib')

modelo = cargar_modelo()

# 3. Cargar los datos SOLO para configurar los mínimos y máximos de los botones
@st.cache_data
def cargar_datos_interfaz():
    datos = pd.read_csv("Cancer_Data.csv")
    # Limpiamos las columnas que no usa el modelo
    columnas_basura = ['id', 'Unnamed: 32', 'perimeter_mean', 'area_mean', 'perimeter_worst', 'area_worst', 'perimeter_se', 'area_se', 'diagnosis']
    columnas_utiles = [col for col in datos.columns if col not in columnas_basura]
    return datos, columnas_utiles

datos_completos, columnas = cargar_datos_interfaz()

st.divider()

# 4. Crear la barra lateral con los controles (sliders)
st.sidebar.header("Datos de la Biopsia del Paciente")
st.sidebar.write("Desliza los valores para introducir las métricas celulares:")

# Generamos automáticamente un deslizador (slider) por cada variable médica
input_usuario = {}
for col in columnas:
    valor_minimo = float(datos_completos[col].min())
    valor_maximo = float(datos_completos[col].max())
    valor_medio = float(datos_completos[col].mean())
    
    # Creamos el control deslizante
    input_usuario[col] = st.sidebar.slider(col, min_value=valor_minimo, max_value=valor_maximo, value=valor_medio)

# 4. El botón de diagnóstico
if st.button("Realizar Diagnóstico", type="primary"):
    df_paciente = pd.DataFrame([input_usuario])
    
    # Hacemos la predicción
    prediccion = modelo.predict(df_paciente)
    prob_maligno = modelo.predict_proba(df_paciente)[0][1]
    porcentaje = int(prob_maligno * 100)
    
    st.subheader("Resultado del Análisis:")
    
    # Barra de progreso visual
    st.write("Nivel de riesgo de malignidad:")
    st.progress(porcentaje)
    
    # Lógica del semáforo clínico
    if prob_maligno >= 0.75:
        st.error(f"**ALERTA ROJA (Confianza: {porcentaje}%):** El modelo clasifica el tumor como MALIGNO con alta seguridad. Se requiere intervención médica inmediata.")
    elif prob_maligno <= 0.25:
        st.success(f"**ZONA VERDE (Confianza Benigna: {100 - porcentaje}%):** El modelo clasifica el tumor como BENIGNO con alta seguridad.")
    else:
        st.warning(f"**ZONA AMARILLA (Riesgo: {porcentaje}%):** El modelo detecta patrones contradictorios. Diagnóstico incierto (requiere evaluación humana y pruebas adicionales).")
