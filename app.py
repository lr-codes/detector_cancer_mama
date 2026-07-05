import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

st.set_page_config(page_title="Detector de Cáncer de Mama", layout="centered")

st.title("Sistema de Diagnóstico de Cáncer de Mama")
st.write("Esta aplicación utiliza un modelo de Machine Learning (Bosque Aleatorio) auditado con una precisión del **95.6%** para predecir si un tumor celular es benigno o maligno.")

# Llamar al modelo una única vez
@st.cache_resource
def preparar_inteligencia_artificial():
    datos = pd.read_csv("Cancer_Data.csv")
    datos = datos.drop(columns=['id', 'Unnamed: 32', 'perimeter_mean', 'area_mean', 'perimeter_worst', 'area_worst', 'perimeter_se', 'area_se'], errors='ignore')
    datos['diagnosis'] = datos['diagnosis'].map({'M': 1 , 'B': 0})
    
    X = datos.drop(columns=['diagnosis'])
    y = datos['diagnosis']
    
    # Entrenar con tus hiperparámetros ganadores del entrenamiento previo   
    modelo = RandomForestClassifier(n_estimators=50, max_depth=10, min_samples_leaf=1, random_state=42)
    modelo.fit(X, y)
    
    return modelo, X.columns, datos

modelo, columnas, datos_completos = preparar_inteligencia_artificial()

# 3. Crear la barra lateral con los controles (sliders)
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