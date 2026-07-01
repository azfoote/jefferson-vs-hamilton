#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 14 14:07:32 2026

@author: aaronfoote
"""

from flask import Flask, request, render_template
import os

# Reduce TensorFlow runtime noise (incl. benign OUT_OF_RANGE end-of-sequence notices)
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # 0=all, 1=INFO, 2=WARNING, 3=ERROR

import numpy as np

try:
    import tensorflow as tf
except ModuleNotFoundError:
    tf = None

app = Flask(__name__)

MODEL_LOAD_ERROR = None
MODEL_IMPORT_ERROR = None
VECTORIZER_ERROR = None
text_vectorization = None

if tf is None:
    MODEL_IMPORT_ERROR = (
        "TensorFlow is not installed in your current Python environment. "
        "Install it (and restart the app) to enable classification."
    )
    model = None
else:
    try:
        # Load model ONCE
        model = tf.keras.models.load_model("classification_model.keras")
    except Exception as e:
        MODEL_LOAD_ERROR = f"Failed to load classification_model.keras: {type(e).__name__}: {e}"
        model = None


def _vectorizer_vocab_path():
    return os.path.join(os.path.dirname(__file__), "vectorizer_vocabulary.txt")


def _read_vocab_file(path: str):
    with open(path, "r", encoding="utf-8") as f:
        # Preserve empty-string tokens by only stripping the newline.
        return [line.rstrip("\n") for line in f]


def _write_vocab_file(path: str, vocab):
    with open(path, "w", encoding="utf-8") as f:
        for token in vocab:
            f.write(f"{token}\n")


def _find_training_dir():
    configured = os.environ.get("TRAIN_DATA_DIR")
    candidates = []
    if configured:
        candidates.append(configured)

    # If you have train/ inside this repo
    candidates.append(os.path.join(os.path.dirname(__file__), "train"))

    for c in candidates:
        if os.path.isdir(c):
            return c
    return None


def _init_text_vectorizer():
    """
    Your saved model expects a (batch, 20000) float vector.
    We recreate the training TextVectorization layer (ngrams=2, max_tokens=20000, output_mode=multi_hot)
    so uploaded text can be converted into that 20k-dimensional input.
    """
    global text_vectorization, VECTORIZER_ERROR

    if tf is None or model is None:
        return

    try:
        from tensorflow.keras.layers import TextVectorization

        input_dim = int(model.input_shape[-1])
        if input_dim <= 0:
            raise ValueError(f"Invalid model input dimension: {model.input_shape}")

        text_vectorization = TextVectorization(
            ngrams=2,
            max_tokens=input_dim,
            output_mode="multi_hot",
        )

        vocab_path = _vectorizer_vocab_path()
        if os.path.exists(vocab_path):
            vocab = _read_vocab_file(vocab_path)
            text_vectorization.set_vocabulary(vocab)
            return

        train_dir = _find_training_dir()
        if not train_dir:
            raise FileNotFoundError(
                "Vectorizer vocabulary not found. Create it by placing "
                f"`{vocab_path}` (one token per line from TextVectorization.get_vocabulary()), "
                "or keep your original training data directory available so the app can adapt the "
                "TextVectorization layer automatically."
            )

        ds = tf.keras.utils.text_dataset_from_directory(
            train_dir,
            batch_size=32,
            shuffle=False,
        )
        text_only = ds.map(lambda x, y: x)
        text_vectorization.adapt(text_only)
        _write_vocab_file(vocab_path, text_vectorization.get_vocabulary())

    except Exception as e:
        VECTORIZER_ERROR = f"Failed to initialize TextVectorization: {type(e).__name__}: {e}"
        text_vectorization = None


# Initialize vectorizer on startup (if model is available)
_init_text_vectorizer()

if tf is not None:
    try:
        tf.get_logger().setLevel("ERROR")
    except Exception:
        pass


def _model_ready_error():
    if MODEL_IMPORT_ERROR:
        return MODEL_IMPORT_ERROR
    if MODEL_LOAD_ERROR:
        return MODEL_LOAD_ERROR
    if VECTORIZER_ERROR:
        return VECTORIZER_ERROR
    if model is None:
        return "Model is not available."
    if text_vectorization is None:
        return "Text vectorizer is not available."
    return None


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html", error=_model_ready_error())

@app.route("/predict", methods=["POST"])
def predict():
    ready_error = _model_ready_error()
    if ready_error:
        return render_template("index.html", error=ready_error), 500

    uploaded_file = request.files.get("file")

    if not uploaded_file:
        return render_template("index.html", error="No file uploaded"), 400

    # Basic file-type guard (the browser UI already restricts to .txt)
    if uploaded_file.filename and not uploaded_file.filename.lower().endswith(".txt"):
        return render_template("index.html", error="Please upload a .txt file"), 400

    try:
        text = uploaded_file.read().decode("utf-8")
    except UnicodeDecodeError:
        return render_template("index.html", error="File must be UTF-8 encoded text"), 400

    if not text.strip():
        return render_template("index.html", error="File was empty (or whitespace only)"), 400

    try:
        # Convert raw text -> 20,000-dim multi-hot bigram vector -> float32
        x_vec = text_vectorization(tf.constant([text]))
        x_vec = tf.cast(x_vec, tf.float32)
        preds = model.predict(x_vec, verbose=0)
    except Exception as e:
        # This is where shape/dtype mismatches typically surface as ValueError.
        return (
            render_template(
                "index.html",
                error=(
                    "Prediction failed. This usually means the model expects a different input "
                    f"shape/dtype than plain text. Details: {type(e).__name__}: {e}"
                ),
            ),
            500,
        )

    # Binary classifier (sigmoid output)
    score = float(np.squeeze(preds))  # probability for the positive class
    # With text_dataset_from_directory, class indices are alphabetical, so score is typically P(class_1),
    # which is usually Jefferson if folders are {hamilton, jefferson}.
    label = "Thomas Jefferson" if score >= 0.5 else "Alexander Hamilton"
    predicted_confidence = score if score >= 0.5 else (1.0 - score)

    return render_template(
        "index.html",
        prediction=label,
        confidence=round(predicted_confidence, 3),
        error=None,
    )

if __name__ == "__main__":
    app.run(debug=True)