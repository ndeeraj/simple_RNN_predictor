# backend web service - flask app
import os

import torch
from flask import Flask, request

from pathlib import Path
from lstmregr import LSTM, tune_model

app = Flask(__name__)
MODEL = None


def create_app():
    """
    This method is the entry point for the flask application.
    Call this method to create the app.

    :return: flask application instance.
    """
    global MODEL
    MODEL = LSTM(num_layers=2, hidden_layer_size=16)
    project_root = Path().resolve()
    weights = 'weights_only_2_16_15.pth'
    weights_file_path = os.path.join(project_root, weights)

    MODEL.to(dtype=torch.double)
    data_dir = 'data'
    data_file = 'data_daily.csv'
    path_to_file = os.path.join(project_root, data_dir, data_file)

    if os.path.isfile(weights_file_path):
        MODEL.load_state_dict(torch.load(weights_file_path))
        MODEL.load_data(path_to_file)
    else:
        tune_model(MODEL, path_to_file)

    return app


@app.post('/predict_with_data')
def predict_with_data():
    """
    Makes the predictions for the requested month and date using the previous data passed in the
    data json
    For a successful request, the data json should follow below constraints on the keys:
    "year": (int) year > 2023
    "month": (int) 1-12
    "prev_data": (list of int) integers in a list of size 15 (window length used by the RNN)

    :return:
    on success, the response will have the data with,
        key "results": contains a dict with keys representing the dates for which
        the predictions were made and values representing predicted values for the date.
    on error, returns data with error message in "error" key.
    """
    global MODEL
    data = request.get_json()

    try:
        out = MODEL.predict_for_month_with_prev(data["prev_data"], data["month"], data["year"])
        response = {"results": out}
        return response
    except Exception as exp:
        response = {"error": f"error while predicting with previous data : {exp}"}
        return response


@app.post('/predict_yy_mm')
def predict_yy_mm():
    """
    Makes the predictions for the requested month and date in the data json
    For a successful request, the data json should follow below constraints on the keys:
    "year": (int) year 2023 or 2024
    "month": (int) 1-12

    :return:
    on success, the response will have the data with,
        key "results": contains a dict with keys representing the dates for which
        the predictions were made and values representing predicted values for the date.
    on error, returns data with error message in "error" key.
    """
    global MODEL
    data = request.get_json()

    try:
        out = MODEL.predict_for_month(data["month"], data["year"])
        response = {"results": out}
        return response
    except Exception as exp:
        response = {"error": f"error while predicting with previous data : {exp}"}
        return response

