# Jefferson vs Hamilton Classifier

A Flask web application that classifies uploaded `.txt` documents as more likely written by **Thomas Jefferson** or **Alexander Hamilton**.

The app uses:
- A trained Keras binary classification model (`classification_model.keras`)
- A saved `TextVectorization` vocabulary (`vectorizer_vocabulary.txt`)
- A simple web interface for file upload and prediction display

## Project structure

- `app.py` - Flask app and prediction pipeline
- `templates/index.html` - HTML UI
- `static/css/style.css` - styling
- `static/images/` - Jefferson and Hamilton portraits
- `classification_model.keras` - trained model file
- `vectorizer_vocabulary.txt` - vocabulary used to vectorize text input
- `binary_bigram_model_development.py` - training script
- `file_extraction.py` - data collection/preparation script

## Quick start

1. Clone the repo and enter the project directory.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run the app:

```bash
python app.py
```

4. Open your browser to:

`http://127.0.0.1:5000`

5. Upload a UTF-8 `.txt` file and click **Analyze Document**.

## How prediction works (simple version)

1. You upload a text file.
2. The app reads it as text.
3. The text is converted into a 20,000-feature multi-hot bigram vector using `TextVectorization`.
4. That numeric vector is passed to the Keras model.
5. The model returns a score and the app displays the predicted author + confidence.

## Results

On a held-out test set (balanced Jefferson / Hamilton documents):

| Metric    | Score  |
|-----------|--------|
| Accuracy  | 99.0%  |
| AUC       | 99.8%  |
| Precision | 99.2%  |
| Recall    | 98.9%  |

- The model performs best on longer passages and can be less stable on very short text samples.
- Output is probabilistic and should be interpreted as a likelihood estimate, not definitive proof of authorship.

## Portable paths and environment variables

This project avoids machine-specific absolute paths.

- `app.py` optionally reads training data from:
  - `TRAIN_DATA_DIR` (if set), otherwise `./train`
- training/data scripts read from:
  - `JH_DATA_DIR` (if set), otherwise `./data`

Examples:

```bash
export TRAIN_DATA_DIR="/path/to/train"
export JH_DATA_DIR="/path/to/data"
```

## Notes

- The app expects `classification_model.keras` and `vectorizer_vocabulary.txt` in the repo root.
- This is a historical text classification experiment and should be treated as probabilistic, not definitive authorship proof.


