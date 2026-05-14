import pandas as pd
import numpy as np
import joblib 
import json
import os
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

# Cargar los datos 
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
df = pd.read_csv(os.path.join(BASE_DIR, '..', 'data', 'googleplaystore_preprocesado.csv'), encoding='latin-1')

#separar características y variable objetivo
features = ['Rating', 'Reviews', 'Size', 'Price',
            'Type_num', 'Category_num', 'ContentRating_num']

X = df[features]
y = df['Exito']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

# modelo esemble(HARD VOTING)

rf  = RandomForestClassifier(n_estimators=100, random_state=42)
gb  = GradientBoostingClassifier(n_estimators=100, random_state=42)
knn = KNeighborsClassifier(n_neighbors=5)

ensemble = VotingClassifier(
    estimators=[('rf', rf), ('gb', gb), ('knn', knn)],
    voting='hard'
)

print("\nEntrenando modelo... (puede tomar 1-2 minutos)")
ensemble.fit(X_train, y_train)

# Evaluar el modelo

y_pred = ensemble.predict(X_test)
acc = accuracy_score(y_test, y_pred)

print(f"\n✅ Accuracy: {acc * 100:.2f}%")
print("\nReporte completo:")
print(classification_report(y_test, y_pred,
                             target_names=['No Exitosa', 'Exitosa']))

# Guardar el modelo y los encoders

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_dir = os.path.join(BASE_DIR, '..', 'model')
os.makedirs(model_dir, exist_ok=True)

ruta_modelo   = os.path.join(model_dir, 'modelo_ensemble.pkl')
ruta_encoders = os.path.join(model_dir, 'encoders.json')

joblib.dump(ensemble, ruta_modelo)

# Generar los maps desde el CSV preprocesado
category_map = {str(v): int(v) for v in df['Category_num'].unique()}
content_map  = {str(v): int(v) for v in df['ContentRating_num'].unique()}

with open(ruta_encoders, 'w') as f:
    json.dump({'category': category_map, 'content': content_map}, f)

print(f"\n Modelo guardado en: {ruta_modelo}")
print(f" Encoders guardados en: {ruta_encoders}")
