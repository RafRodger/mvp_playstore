# Predictor de Éxito en Google Play Store

Aplicativo web basado en inteligencia artificial que predice el éxito comercial de una aplicación móvil Android, desarrollado como proyecto final de la asignatura de Inteligencia Artificial — Ingeniería de Sistemas.

---

## Integrantes

- Mario Uparela  
- Rafael Rodger  
- Justin Castro  

---

## Descripción

Este MVP utiliza un modelo de **Ensemble Learning (Hard Voting)** que combina tres algoritmos de clasificación:

- Random Forest
- Gradient Boosting
- K-Nearest Neighbors (KNN)

El modelo fue entrenado con el dataset de Google Play Store de Kaggle (más de 10,000 aplicaciones) y clasifica una app como **exitosa** si supera las 100,000 instalaciones.

---

##  Estructura del proyecto

```
mvp_playstore/
├── data/
│   └── googleplaystore_preprocesado.csv
├── model/
│   ├── modelo_ensemble.pkl
│   └── encoders.json
├── notebooks/
│   ├── pre_procesamiento.ipynb
│   └── entrenar_modelo.py
├── app/
│   ├── app.py
│   └── templates/
│       └── index.html
├── requirements.txt
└── README.md
```

---

## Preprocesamiento de datos

El preprocesamiento se realizó en el notebook `notebooks/pre_procesamiento.ipynb` siguiendo estos pasos:

**1. Limpieza de la columna `Installs`**  
Se eliminaron los caracteres `+` y `,` y se convirtió a valor numérico.

**2. Limpieza de la columna `Size`**  
Se creó una función que convierte los valores con sufijo `M` (megabytes) a kilobytes multiplicando por 1024, y los valores con sufijo `k` se dejan como kilobytes directamente. Los valores `Varies with device` se tratan como nulos.

**3. Limpieza de `Price` y `Rating`**  
Se eliminó el símbolo `$` de la columna de precio y se convirtió a numérico. Los valores nulos en precio se rellenaron con `0`. El rating se convirtió a numérico eliminando entradas inválidas.

**4. Creación de la variable objetivo**  
Se creó la columna `Exito` con valor `1` si la app tiene 100,000 o más instalaciones, y `0` en caso contrario.

**5. Eliminación de nulos**  
Se eliminaron las filas con valores nulos en `Rating`, `Size` e `Installs`.

**6. Codificación de variables categóricas**  
- `Type` → `Type_num`: `1` si la app es gratuita, `0` si es de pago.  
- `Category` → `Category_num`: codificación numérica con `pd.factorize`.  
- `Content Rating` → `ContentRating_num`: codificación numérica con `pd.factorize`.

El dataset resultante se exportó como `googleplaystore_preprocesado.csv` en la carpeta `data/`.

Para reproducir el preprocesamiento ejecuta el notebook:

```bash
cd notebooks
jupyter notebook pre_procesamiento.ipynb
```

---

## Instalación y uso

### 1. Clona el repositorio

```bash
git clone https://github.com/RafRodger/mvp_playstore.git
cd mvp_playstore
```

### 2. Crea el entorno virtual e instala dependencias

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Mac/Linux
source venv/bin/activate

pip install -r requirements.txt
```

### 3. Preprocesa los datos

```bash
cd notebooks
jupyter notebook pre_procesamiento.ipynb
```

### 4. Entrena el modelo

```bash
python entrenar_modelo.py
```

Esto genera los archivos `modelo_ensemble.pkl` y `encoders.json` dentro de la carpeta `model/`.

### 5. Corre la aplicación

```bash
cd app
python app.py
```

Abre tu navegador en `http://127.0.0.1:5000`

---

## Variables del modelo

| Variable | Descripción |
|---|---|
| Rating | Calificación de la app (0–5) |
| Reviews | Número de reseñas |
| Size | Tamaño del APK en KB |
| Price | Precio en dólares (0 si es gratis) |
| Type | Gratis o de pago |
| Category | Categoría de la app en Play Store |
| Content Rating | Clasificación de contenido |

---

## Resultados del modelo

| Métrica | Valor |
|---|---|
| Accuracy | ~96% |
| Algoritmo base | Hard Voting Ensemble |
| Dataset | Google Play Store — Kaggle (Gupta, 2019) |
| Split | 75% entrenamiento / 25% prueba |

---

## Tecnologías usadas

- **Python 3.x**
- **Flask** — servidor web
- **scikit-learn** — modelo de machine learning
- **pandas / numpy** — procesamiento de datos
- **joblib** — serialización del modelo
- **HTML / CSS / JavaScript** — interfaz web

---

## Referencias

- Gupta, L. (2019). *Google Play Store Apps*. Kaggle.
- Saleem, A., et al. (2024). *Predicting Mobile App Success Using a Robust Hard Voting Ensemble Learning Approach.*
- Pattanaik, P., & Nagpal, D. (2023). *Comparison of machine learning algorithms used to catalog Google Appstore.*
- Zuhir, N., et al. (2024). *Beyond Stars: Bridging the Gap Between Ratings and Review Sentiment with LLM.*

---

## Notas

- El archivo `.pkl` del modelo no se incluye en el repositorio por su tamaño. Debes generarlo ejecutando `entrenar_modelo.py`.
- El dataset original tampoco se incluye. Descárgalo desde [Kaggle](https://www.kaggle.com/datasets/lava18/google-play-store-apps) y colócalo en la carpeta `data/` antes de ejecutar el preprocesamiento.
