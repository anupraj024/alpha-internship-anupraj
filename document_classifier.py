#!/usr/bin/env python3
"""Intelligent Document Classification System using TF-IDF and Naive Bayes."""
import argparse
import json
import re
import sys
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, precision_recall_fscore_support)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline


BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs"


def clean_text(text):
    """Normalize text while preserving meaningful alphanumeric terms."""
    text = str(text).lower()
    text = re.sub(r"http\\S+|www\\.\\S+", " ", text)
    text = re.sub(r"[^a-z0-9\\s]", " ", text)
    text = re.sub(r"\\s+", " ", text).strip()
    return text


def load_dataset(path):
    df = pd.read_csv(path)
    required = {"text", "label"}
    if not required.issubset(df.columns):
        raise ValueError("CSV must contain 'text' and 'label' columns.")
    df = df.dropna(subset=["text", "label"]).copy()
    df["text"] = df["text"].map(clean_text)
    df["label"] = df["label"].astype(str).str.strip()
    df = df[(df["text"] != "") & (df["label"] != "")]
    if df["label"].nunique() < 2:
        raise ValueError("Dataset must contain at least two distinct labels.")
    return df


def train(data_path, test_size=0.25, random_state=42):
    df = load_dataset(data_path)
    counts = df["label"].value_counts()
    if counts.min() < 2:
        raise ValueError("Each category needs at least two samples for a stratified split.")

    x_train, x_test, y_train, y_test = train_test_split(
        df["text"], df["label"], test_size=test_size,
        random_state=random_state, stratify=df["label"]
    )

    model = Pipeline([
        ("tfidf", TfidfVectorizer(
            stop_words="english",
            ngram_range=(1, 2),
            max_features=12000,
            sublinear_tf=True
        )),
        ("classifier", MultinomialNB(alpha=0.5))
    ])
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)

    accuracy = accuracy_score(y_test, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, predictions, average="weighted", zero_division=0
    )
    labels = sorted(df["label"].unique())
    report = {
        "project": "Intelligent Document Classification System",
        "algorithm": "Multinomial Naive Bayes",
        "feature_extraction": "TF-IDF (unigrams and bigrams)",
        "dataset": str(Path(data_path)),
        "total_documents": int(len(df)),
        "training_documents": int(len(x_train)),
        "testing_documents": int(len(x_test)),
        "categories": labels,
        "metrics": {
            "accuracy": round(float(accuracy), 4),
            "weighted_precision": round(float(precision), 4),
            "weighted_recall": round(float(recall), 4),
            "weighted_f1_score": round(float(f1), 4)
        },
        "classification_report": classification_report(
            y_test, predictions, output_dict=True, zero_division=0
        ),
        "confusion_matrix": {
            "labels": labels,
            "matrix": confusion_matrix(y_test, predictions, labels=labels).tolist()
        },
        "note": "Metrics are from a held-out split of the included small illustrative dataset."
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, OUTPUT_DIR / "document_classifier.joblib")
    (OUTPUT_DIR / "evaluation_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )

    print("\\nINTELLIGENT DOCUMENT CLASSIFICATION")
    print("=" * 39)
    print(f"Documents: {len(df)} | Train: {len(x_train)} | Test: {len(x_test)}")
    print(f"Categories: {', '.join(labels)}")
    print("\\nEvaluation metrics (weighted):")
    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1-score : {f1:.4f}")
    print("\\nDetailed classification report:")
    print(classification_report(y_test, predictions, zero_division=0))
    print(f"Saved model: {OUTPUT_DIR / 'document_classifier.joblib'}")
    print(f"Saved report: {OUTPUT_DIR / 'evaluation_report.json'}")
    return report


def predict(text=None, file_path=None):
    model_path = OUTPUT_DIR / "document_classifier.joblib"
    if not model_path.exists():
        raise FileNotFoundError("Trained model not found. Run the train command first.")
    if file_path:
        text = Path(file_path).read_text(encoding="utf-8")
    if not text or not clean_text(text):
        raise ValueError("Provide non-empty text using --text or --file.")
    model = joblib.load(model_path)
    cleaned = clean_text(text)
    category = model.predict([cleaned])[0]
    probabilities = model.predict_proba([cleaned])[0]
    classes = model.named_steps["classifier"].classes_
    ranked = sorted(zip(classes, probabilities), key=lambda item: item[1], reverse=True)
    print("\\nDOCUMENT CLASSIFICATION RESULT")
    print("=" * 31)
    print(f"Predicted category: {category}")
    print("Class confidence scores:")
    for label, score in ranked:
        print(f"  {label}: {score:.4f}")
    return category


def main():
    parser = argparse.ArgumentParser(description="NLP-based document classification.")
    sub = parser.add_subparsers(dest="command", required=True)
    t = sub.add_parser("train", help="Preprocess data, train model, and evaluate.")
    t.add_argument("--data", type=Path, default=BASE_DIR / "data" / "documents.csv")
    t.add_argument("--test-size", type=float, default=0.25)
    p = sub.add_parser("predict", help="Predict category for new text.")
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--text")
    group.add_argument("--file", type=Path)
    args = parser.parse_args()
    try:
        if args.command == "train":
            if not 0.1 <= args.test_size <= 0.5:
                raise ValueError("test-size must be between 0.1 and 0.5.")
            train(args.data, args.test_size)
        else:
            predict(text=args.text, file_path=args.file)
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
