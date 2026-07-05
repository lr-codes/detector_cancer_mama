import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import GridSearchCV, cross_val_score
import joblib

import matplotlib.pyplot as plt
import seaborn as sns


# =====================================
#   Corrección y separación de Datos
# =====================================

datos = pd.read_csv("Cancer_Data.csv")

datos = datos.drop(columns=['id'])
datos = datos.drop(columns=['Unnamed: 32'])
datos = datos.drop(columns=['perimeter_mean', 'area_mean', 'perimeter_worst', 'area_worst', 'perimeter_se', 'area_se'])

datos['diagnosis'] = datos['diagnosis'].map({'M': 1 , 'B':0})

X = datos.drop(columns=['diagnosis'])
y = datos['diagnosis']

# =========================================
#   Entrenamiento del modelo (GridSearch)
# =========================================

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


parametros = {
    'n_estimators':[50, 100, 200, 400], 
    'max_depth':[3, 5, 10, None],
    'min_samples_leaf':[1, 2, 4, 8]
}
modelo = RandomForestClassifier(n_estimators=100, random_state=42)



buscador = GridSearchCV(estimator=modelo, param_grid=parametros, cv=5, n_jobs=-1, scoring='accuracy')
print("--- BUSCANDO MODELO ADECUADO ---")
buscador.fit(X_train, y_train)

mejor_modelo = buscador.best_estimator_
print("--- BUSQUEDA FINALIZADA ---")
print(buscador.best_params_)

# ===============================================================
#   Evaluación del modelo e importancias con umbral arbittrario
# ===============================================================

probabilidades = mejor_modelo.predict_proba(X_test)[:, 1]

# 2. Definimos nuestro nuevo umbral médico 
umbral_medico = 0.40

# 3. Aplicamos la regla: Si la probabilidad supera el umbral, es 1
predicciones = (probabilidades >= umbral_medico).astype(int)

print("\n--- INFORME MÉDICO DE ALTA SENSIBILIDAD ---")
print(classification_report(y_test, predicciones))

precision = accuracy_score(y_test, predicciones)
print(f"La precisión es de {precision*100:.2f}%\n")


# ====================================
#   Prueba mediante Cross Validation
# ====================================

nota = cross_val_score(mejor_modelo, X, y, cv=5, scoring='accuracy')

print("\n--- RESULTADOS DE LA VALIDACIÓN CRUZADA ---")
print(f"Notas de los 5 exámenes separados: {nota}")

precision_media = nota.mean()
print(f"PRECISIÓN MEDIA REAL DEL MODELO: {precision_media * 100:.2f}%")

# --- GRAFICA ---
importancia_emp = pd.Series(mejor_modelo.feature_importances_, index=X_train.columns)
importancia_ord = importancia_emp.sort_values(ascending=False)

plt.figure(figsize=(10,6))

sns.barplot(x=importancia_ord.values, 
            y=importancia_ord.index, 
            hue=importancia_ord.index, 
            palette="viridis", 
            legend=False)
plt.title("Factores de riesgo mas importantes", fontsize=15, pad=15)
plt.xlabel("Nivel de importancia", fontsize=13)
plt.ylabel("Variables", fontsize=13)
plt.tight_layout()
plt.show()

# --- EXPORTACION DEL MODELO ---
joblib.dump(mejor_modelo, 'modelo_cancer.joblib')
print("¡Modelo exportado con éxito!")

