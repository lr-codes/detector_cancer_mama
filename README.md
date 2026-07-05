# Sistema de Diagnóstico de Cáncer de Mama (Machine Learning)

Esta aplicación web utiliza un modelo de inteligencia artificial (Bosque Aleatorio / Random Forest) para clasificar tumores mamarios como **Benignos** o **Malignos** basándose en métricas celulares extraídas de biopsias (Punción Aspirativa con Aguja Fina - PAAF).

El modelo ha sido entrenado y rigurosamente auditado mediante validación cruzada (Cross-Validation), alcanzando una **precisión real del 95.6%**.

---

## Cómo usar la aplicación

Puedes probar la aplicación directamente desde el navegador a través del siguiente enlace:
**[\[Enlace a la Aplicación en Vivo\]](https://cancer-mama.streamlit.app/)**

### Ejecución en local (Para desarrolladores)
Si deseas ejecutar este proyecto en tu propia máquina:
1. Clona este repositorio.
2. Instala las dependencias requeridas:
   ```bash
   pip install -r requirements.txt
3. Lanza la aplicación mediante Streamlit:  
   ```bash
   python -m streamlit run app.py
## Diccionario de Datos
Los datos introducidos en la aplicación corresponden a medidas microscópicas de los núcleos de las células mamarias. Para cada característica base, la herramienta evalúa tres dimensiones matemáticas:

* **Mean (Media):** Valor promedio de los núcleos analizados en la muestra.
* **SE (Error Estándar):** Variabilidad o error estándar en la medida.
* **Worst (Peor):** El promedio de los 3 valores mas extremo/grandes encontrados, crucial para detectar células anómalas.

Diferentes Características:

* **Radius (Radio):** Distancia desde el centro del núcleo hasta su perímetro. Los núcleos malignos tienden a ser más grandes.
* **Texture (Textura):** Variación en la escala de grises de la imagen microscópica. Mide como de "rugoso" parece el interior del núcleo.
* **Smoothness (Suavidad):** Mide la variación local en las longitudes del radio. Un contorno irregular suele ser indicativo de malignidad.
* **Compactness (Compacidad):** Relación matemática entre el perímetro y el área `(Perímetro² / Área - 1.0)`. Evalúa qué tan "denso" es el núcleo. 
* **Concavity (Concavidad):** Severidad o profundidad de las porciones cóncavas (hendiduras) en el contorno del núcleo.
* **Concave Points (Puntos Cóncavos):** Cantidad total de hendiduras en el contorno. Es uno de los predictores de riesgo más fuertes.
* **Symmetry (Simetría):** Diferencias de forma al partir el núcleo por la mitad. Las células cancerígenas suelen ser altamente asimétricas.
* **Fractal Dimension (Dimensión Fractal):** Medida de la complejidad del borde del núcleo `("Aproximación de la línea costera" - 1)`.
