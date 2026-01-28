#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 14 14:07:32 2026

@author: aaronfoote
"""

from flask import Flask, request, render_template
import tensorflow as tf
import numpy as np

app = Flask(__name__)

# Load model ONCE
model = tf.keras.models.load_model("classification_model.keras")

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    uploaded_file = request.files.get("file")

    if not uploaded_file:
        return "No file uploaded", 400

    try:
        text = uploaded_file.read().decode("utf-8")
    except UnicodeDecodeError:
        return "File must be UTF-8 encoded text", 400

    # Model expects a batch of strings
    inputs = np.array([text])

    preds = model.predict(inputs)

    # Binary classifier (sigmoid output)
    score = float(preds[0][0])
    label = "State A" if score >= 0.5 else "State B"

    return render_template(
        "index.html",
        prediction=label,
        confidence=round(score, 3)
    )

if __name__ == "__main__":
    app.run(debug=True)