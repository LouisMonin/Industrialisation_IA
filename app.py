from flask import Flask, request, jsonify
import pickle
import pandas as pd

# Charger les modèles et préprocesseurs
with open('model_freq.pkl', 'rb') as f: # en attendant de le faire
    model_freq = pickle.load(f)

with open('preproc_freq.pkl', 'rb') as f: # en attendant de le faire
    preproc_freq = pickle.load(f)

with open('model_montant.pkl', 'rb') as f:
    model_montant = pickle.load(f)

with open('preproc_montant.pkl', 'rb') as f: # en attendant de le faire
    preproc_montant = pickle.load(f)

# Créer l’application
app = Flask(__name__)

# Route 1 : Health check
@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok'}), 200


# Route 2 : Prédiction fréquence
@app.route('/predict/freq', methods=['POST'])
def predict_freq():
    data = request.get_json()
    df = pd.DataFrame([data])
    X = preproc_freq.transform(df)
    y_pred = model_freq.predict(X)
    return jsonify({'prediction_freq': float(y_pred[0])})


# Route 3 : Prédiction montant
@app.route('/predict/montant', methods=['POST'])
def predict_montant():
    data = request.get_json()
    df = pd.DataFrame([data])
    X = preproc_montant.transform(df)
    y_pred = model_montant.predict(X)
    return jsonify({'prediction_montant': float(y_pred[0])})


# Route 4 : Prédiction globale
@app.route('/predict/global', methods=['POST'])
def predict_global():
    data = request.get_json()
    df = pd.DataFrame([data])
    X_freq = preproc_freq.transform(df)
    X_montant = preproc_montant.transform(df)
    freq = model_freq.predict(X_freq)[0]
    montant = model_montant.predict(X_montant)[0]
    return jsonify({
        'prediction_freq': float(freq),
        'prediction_montant': float(montant),
        'prediction_global': float(freq * montant)
    })


if __name__ == '__main__':
    app.run(debug=True)
