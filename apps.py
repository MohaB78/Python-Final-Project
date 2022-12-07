import pickle
from urllib import request

import pandas as pd
from django.apps import AppConfig
import os
import joblib
from django.apps import AppConfig
from django.conf import settings
from flask import jsonify, Flask

from models import rf_model, X_test

app = Flask(__name__)
print("Loading : Done")

model = pickle.load(open('model.pkl', "rb"))


@app.route('/predict/', methods=['POST'])
def predict():
    json_ = request.json
    query_df = pd.DataFrame(json_)
    prediction = rf_model.predict(X_test)
    return jsonify('Prediction', list(prediction))


app.run(host='127.0.0.1', port=8080, debug=False)

if __name__ == "__main__":
    app.run(debug=True)
