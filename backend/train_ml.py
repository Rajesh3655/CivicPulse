from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import json
import os
import threading
import time

from dotenv import load_dotenv
import joblib
from pymongo import MongoClient
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split


load_dotenv()

BASE_DIR = os.path.dirname(__file__)
MODEL_DIR = os.path.join(BASE_DIR, "models")
REPORT_PATH = os.path.join(MODEL_DIR, "training_report.json")
AUTO_RETRAIN_MIN_FEEDBACK = int(os.environ.get("AUTO_RETRAIN_MIN_FEEDBACK", "5"))
AUTO_RETRAIN_COOLDOWN_SECONDS = int(os.environ.get("AUTO_RETRAIN_COOLDOWN_SECONDS", "300"))

_training_lock = threading.Lock()
_last_auto_train_at = 0.0


def normalize_text(value: str | None) -> str:
    return " ".join((value or "").split()).strip()


def ensure_model_dir() -> None:
    os.makedirs(MODEL_DIR, exist_ok=True)


def connect_db():
    mongo_uri = os.environ.get("MONGO_URI", "mongodb://localhost:27017/")
    client = MongoClient(mongo_uri)
    return client["civic_issues_db"]


def fetch_labeled_issues(db) -> list[dict]:
    issues = list(
        db.issues.find(
            {},
            {
                "title": 1,
                "description": 1,
                "category": 1,
                "priority": 1,
                "corrected_category": 1,
                "corrected_priority": 1,
                "department_name": 1,
            },
        )
    )
    labeled = []
    for issue in issues:
        title = normalize_text(issue.get("title"))
        description = normalize_text(issue.get("description"))
        text = normalize_text(f"{title}. {description}. Department {issue.get('department_name', '')}")
        category = normalize_text(issue.get("corrected_category") or issue.get("category"))
        priority = normalize_text(issue.get("corrected_priority") or issue.get("priority"))
        if not text or not category or not priority:
            continue
        labeled.append({"text": text, "category": category, "priority": priority})
    return labeled


def train_text_model(texts: list[str], labels: list[str]) -> tuple[TfidfVectorizer, LogisticRegression, dict]:
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words="english", min_df=1)
    classifier = LogisticRegression(max_iter=1200, class_weight="balanced")

    if len(set(labels)) < 2 or len(texts) < 6:
        matrix = vectorizer.fit_transform(texts)
        classifier.fit(matrix, labels)
        report = {
            "train_size": len(texts),
            "test_size": 0,
            "accuracy": None,
            "classification_report": {},
            "note": "Dataset too small for a meaningful holdout test; model trained on all available samples.",
        }
        return vectorizer, classifier, report

    stratify = labels if min(Counter(labels).values()) >= 2 else None
    x_train, x_test, y_train, y_test = train_test_split(
        texts,
        labels,
        test_size=0.2,
        random_state=42,
        stratify=stratify,
    )
    x_train_matrix = vectorizer.fit_transform(x_train)
    classifier.fit(x_train_matrix, y_train)

    x_test_matrix = vectorizer.transform(x_test)
    predictions = classifier.predict(x_test_matrix)
    report = {
        "train_size": len(x_train),
        "test_size": len(x_test),
        "accuracy": round(float(accuracy_score(y_test, predictions)), 4),
        "classification_report": classification_report(y_test, predictions, output_dict=True, zero_division=0),
    }
    return vectorizer, classifier, report


def train_models(db) -> dict:
    ensure_model_dir()
    labeled = fetch_labeled_issues(db)
    if len(labeled) < 3:
        raise ValueError("Not enough labeled issues to train models. Add more issues or admin corrections first.")

    texts = [item["text"] for item in labeled]
    categories = [item["category"] for item in labeled]
    priorities = [item["priority"] for item in labeled]

    category_vectorizer, category_model, category_report = train_text_model(texts, categories)
    priority_vectorizer, priority_model, priority_report = train_text_model(
        [f"{category} {text}" for category, text in zip(categories, texts)],
        priorities,
    )

    joblib.dump(category_vectorizer, os.path.join(MODEL_DIR, "category_vectorizer.joblib"))
    joblib.dump(category_model, os.path.join(MODEL_DIR, "category_model.joblib"))
    joblib.dump(priority_vectorizer, os.path.join(MODEL_DIR, "priority_vectorizer.joblib"))
    joblib.dump(priority_model, os.path.join(MODEL_DIR, "priority_model.joblib"))

    report = {
        "trained_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "dataset_size": len(labeled),
        "category_labels": dict(Counter(categories)),
        "priority_labels": dict(Counter(priorities)),
        "category_model": category_report,
        "priority_model": priority_report,
    }
    with open(REPORT_PATH, "w", encoding="utf-8") as file:
        json.dump(report, file, indent=2)
    return report


def should_auto_retrain(db) -> bool:
    feedback_count = db.issues.count_documents({'ai_feedback.0': {'$exists': True}})
    return feedback_count >= AUTO_RETRAIN_MIN_FEEDBACK


def _run_training_job(db) -> None:
    global _last_auto_train_at
    with _training_lock:
        now = time.time()
        if now - _last_auto_train_at < AUTO_RETRAIN_COOLDOWN_SECONDS:
            return
        try:
            train_models(db)
            _last_auto_train_at = time.time()
        except Exception:
            pass


def trigger_training_async(db, *, force: bool = False) -> bool:
    if _training_lock.locked():
        return False
    if not force and not should_auto_retrain(db):
        return False
    thread = threading.Thread(target=_run_training_job, args=(db,), daemon=True)
    thread.start()
    return True


def ensure_model_ready_async(db) -> bool:
    has_trained_model = os.path.exists(os.path.join(MODEL_DIR, "category_model.joblib"))
    if has_trained_model:
        return False
    thread = threading.Thread(target=_run_training_job, args=(db,), daemon=True)
    thread.start()
    return True


def main() -> None:
    db = connect_db()
    report = train_models(db)

    print("Training complete.")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
