# Intelligent Document Classification System

An educational NLP and machine-learning project based on the supplied project brief.

## Project overview
The system classifies text documents into predefined categories. It demonstrates:
- Dataset collection and preprocessing
- Text cleaning and tokenization
- TF-IDF feature extraction
- Supervised AI model training and testing
- Classification using a Multinomial Naive Bayes model
- Evaluation using Accuracy, Precision, Recall, and F1-score
- Prediction for a new document

## Technology
- Python 3.9+
- pandas
- scikit-learn
- joblib
- TF-IDF vectorization
- Multinomial Naive Bayes

## Setup
```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

python -m pip install -r requirements.txt
```

## Train and evaluate
From this project directory:
```bash
python document_classifier.py train --data data/documents.csv
```
The script creates `outputs/model.joblib`, `outputs/vectorizer.joblib`, and `outputs/evaluation_report.json`.

## Classify new text
```bash
python document_classifier.py predict --text "The team won the football match after scoring two goals."
```

## Classify a text file
```bash
python document_classifier.py predict --file sample_input.txt
```

## Dataset
`data/documents.csv` is a small, balanced, illustrative dataset created for demonstrating the workflow. It is not a production dataset and should not be used to make consequential decisions. For stronger results, replace it with a larger, representative, properly licensed dataset.

## Screenshots for submission
Run the training and prediction commands on your computer and take genuine screenshots of:
1. Successful training and evaluation metrics.
2. The generated evaluation report.
3. A sample prediction and predicted category.

The `outputs/` folder includes a report generated when this package was built (where dependencies were available). Re-run training locally to reproduce it. No screenshots are fabricated or included.

## Limitations and responsible use
- The model can only predict categories present in its training data.
- Accuracy depends on dataset size, quality, balance, and similarity to real documents.
- The included dataset is intentionally small and illustrative; evaluation scores may not represent real-world performance.
- Do not use this demonstration to classify confidential, sensitive, or high-impact documents without appropriate validation, privacy safeguards, and human review.
