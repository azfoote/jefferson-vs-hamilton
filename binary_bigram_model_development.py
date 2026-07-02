#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  8 16:06:53 2026

@author: aaronfoote
"""

#The following document contains code for building and training a machine
#learning model for text classification.

#5400 text files written by either Thomas Jefferson or Alexander Hamilton (10800 files total)
#have been split into three folders: 50% training, 20% validation, and 30% testing.

#The code below is assumes a directory structure like the following:
    #...train/
    #......jefferson/
    #......hamilton/
    #...val/
    #......jefferson/
    #......hamilton/
    #...test/
    #......jefferson/
    #......hamilton/
    
#First, tensorflow and keras are used to create a batched dataset of each text
#file and their labels

import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
from tensorflow.keras.layers import TextVectorization

keras.utils.set_random_seed(42)

BATCH_SIZE = 64
MAX_TOKENS = 20000  # Keep 20k for compatibility with app.py input expectations
AUTOTUNE = tf.data.AUTOTUNE

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def _resolve_base_data_dir():
    """
    Find the folder that contains train/, val/, and test/ subfolders.

    Order:
    1. JH_DATA_DIR environment variable (if train/ exists under it)
    2. ./data/train (repo's data/ layout)
    3. ./train (splits copied next to this script)
    4. ../train (splits in the parent project folder, e.g. Jefferson_Hamilton_Classification/)
    """
    explicit = os.environ.get("JH_DATA_DIR")
    if explicit and os.path.isdir(os.path.join(explicit, "train")):
        return explicit

    candidates = [
        os.path.join(_SCRIPT_DIR, "data"),
        _SCRIPT_DIR,
        os.path.join(_SCRIPT_DIR, ".."),
    ]
    for base in candidates:
        base = os.path.abspath(base)
        if os.path.isdir(os.path.join(base, "train")):
            return base

    raise FileNotFoundError(
        "Could not find training data. Expected train/, val/, and test/ under one of:\n"
        f"  - JH_DATA_DIR (set this env var), e.g. export JH_DATA_DIR=\"/path/to/parent\"\n"
        f"  - {_SCRIPT_DIR}/data\n"
        f"  - {_SCRIPT_DIR}\n"
        f"  - {os.path.abspath(os.path.join(_SCRIPT_DIR, '..'))}\n"
        "Put class folders (e.g. Hamilton/, Jefferson/) inside each split."
    )


BASE_DATA_DIR = _resolve_base_data_dir()
print(f"Using data directory: {BASE_DATA_DIR}")

train_ds = keras.utils.text_dataset_from_directory(
    os.path.join(BASE_DATA_DIR, "train"),
    batch_size=BATCH_SIZE,
    shuffle=True,
    seed=42,
)
val_ds = keras.utils.text_dataset_from_directory(
    os.path.join(BASE_DATA_DIR, "val"),
    batch_size=BATCH_SIZE,
    shuffle=False,
)
test_ds = keras.utils.text_dataset_from_directory(
    os.path.join(BASE_DATA_DIR, "test"),
    batch_size=BATCH_SIZE,
    shuffle=False,
)

print("Class names:", train_ds.class_names)

# Add a preprocessing text vectorization layer to encode the most common 20,000
# unigrams+bigrams as sparse numerical vectors.
text_vectorization = TextVectorization(
    ngrams=2,
    max_tokens=MAX_TOKENS,
    output_mode="multi_hot",
)

text_only_train_ds = train_ds.map(lambda x, y: x)
text_vectorization.adapt(text_only_train_ds)

# Export vocabulary so the Flask app can load it (one token per line).
# The app looks for vectorizer_vocabulary.txt in this project directory.
VOCAB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "vectorizer_vocabulary.txt")
vocab = text_vectorization.get_vocabulary()
with open(VOCAB_PATH, "w", encoding="utf-8") as f:
    for token in vocab:
        f.write(f"{token}\n")
print(f"Exported vocabulary ({len(vocab)} tokens) to {VOCAB_PATH}")


def vectorize(ds):
    return (
        ds.map(lambda x, y: (text_vectorization(x), y), num_parallel_calls=AUTOTUNE)
        .cache()
        .prefetch(AUTOTUNE)
    )


binary_2gram_train_ds = vectorize(train_ds)
binary_2gram_val_ds = vectorize(val_ds)
binary_2gram_test_ds = vectorize(test_ds)


def get_model(max_tokens=MAX_TOKENS):
    # A dense model with regularization and dropout layers works well in general with bag of n-grams input such as this.
    inputs = keras.Input(shape=(max_tokens,))
    x = layers.Dense(128, activation="relu", kernel_regularizer=regularizers.l2(1e-4))(inputs)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(64, activation="relu", kernel_regularizer=regularizers.l2(1e-4))(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)

    model = keras.Model(inputs, outputs)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
        metrics=[
            keras.metrics.BinaryAccuracy(name="accuracy"),
            keras.metrics.AUC(name="auc"),
            keras.metrics.Precision(name="precision"),
            keras.metrics.Recall(name="recall"),
        ],
    )
    return model


model = get_model()
model.summary()

callbacks = [
    keras.callbacks.ModelCheckpoint("classification_model.keras", monitor="val_auc", mode="max", save_best_only=True),
    keras.callbacks.EarlyStopping(monitor="val_auc", mode="max", patience=3, restore_best_weights=True),
    keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-5),
]

history = model.fit(
    binary_2gram_train_ds,
    validation_data=binary_2gram_val_ds,
    epochs=20,
    callbacks=callbacks,
)

best_model = keras.models.load_model("classification_model.keras")
results = best_model.evaluate(binary_2gram_test_ds, return_dict=True)
print("Test metrics:", {k: round(v, 4) for k, v in results.items()})
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
