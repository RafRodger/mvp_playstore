from flask import Flask, request, render_template, jsonify
import joblib, json
import os 
import numpy as np

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model = joblib.load(os.path.join(BASE_DIR, '..', 'model', 'modelo_ensemble.pkl'))

CATEGORY_MAP = {
    'ART_AND_DESIGN': 0, 'AUTO_AND_VEHICLES': 1, 'BEAUTY': 2,
    'BOOKS_AND_REFERENCE': 3, 'BUSINESS': 4, 'COMICS': 5,
    'COMMUNICATION': 6, 'DATING': 7, 'EDUCATION': 8,
    'ENTERTAINMENT': 9, 'EVENTS': 10, 'FINANCE': 11,
    'FOOD_AND_DRINK': 12, 'HEALTH_AND_FITNESS': 13, 'HOUSE_AND_HOME': 14,
    'LIBRARIES_AND_DEMO': 15, 'LIFESTYLE': 16, 'GAME': 17,
    'FAMILY': 18, 'MEDICAL': 19, 'SOCIAL': 20, 'SHOPPING': 21,
    'PHOTOGRAPHY': 22, 'SPORTS': 23, 'TRAVEL_AND_LOCAL': 24,
    'TOOLS': 25, 'PERSONALIZATION': 26, 'PRODUCTIVITY': 27,
    'PARENTING': 28, 'WEATHER': 29, 'VIDEO_PLAYERS': 30,
    'NEWS_AND_MAGAZINES': 31, 'MAPS_AND_NAVIGATION': 32
}

CONTENT_MAP = {
    'Everyone': 0, 'Teen': 1, 'Everyone 10+': 2,
    'Mature 17+': 3, 'Adults only 18+': 4, 'Unrated': 5
}

@app.route('/')
def index():
    categories = list(CATEGORY_MAP.keys())
    contents   = list(CONTENT_MAP.keys())
    return render_template('index.html', categories=categories, contents=contents)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.form

    def to_float(val, default=0.0):
        try:
            return float(val)
        except (ValueError, TypeError):
            return default

    features = np.array([[
        to_float(data['rating']),
        to_float(data['reviews']),
        to_float(data['size']),
        to_float(data['price']),
        1 if data['type'] == 'Free' else 0,
        CATEGORY_MAP[data['category']],
        CONTENT_MAP[data['content_rating']],
    ]])

    pred      = model.predict(features)[0]
    resultado = "✅ La app tiene perfil de ÉXITO" if pred == 1 else "❌ La app NO tiene perfil exitoso"

    return jsonify({'resultado': resultado})

if __name__ == '__main__':
    app.run(debug=True)