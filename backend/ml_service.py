from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import math
import json
import os
import re
from typing import Iterable
from PIL import Image, ImageStat
import joblib
from dotenv import load_dotenv

load_dotenv()

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics.pairwise import cosine_similarity
except Exception:  # pragma: no cover - optional runtime dependency fallback
    TfidfVectorizer = None
    LogisticRegression = None
    cosine_similarity = None


DEFAULT_CATEGORY_EXAMPLES = {
    "Pothole": [
        "large pothole in the middle of the road",
        "deep hole on main street causing traffic danger",
        "road surface broken with pothole after rain",
    ],
    "Road Damage": [
        "damaged road surface with cracked asphalt",
        "road is broken and unsafe for vehicles",
        "huge crack in the roadway near junction",
    ],
    "Footpath": [
        "footpath tiles broken and unsafe for walking",
        "sidewalk damaged near bus stop",
        "pedestrian path blocked and cracked",
    ],
    "Street Light": [
        "street light not working at night",
        "broken lamp post causing dark road",
        "no light on the road near the school",
    ],
    "Power Outage": [
        "power outage in the neighborhood",
        "electricity supply failure in our street",
        "current is gone for many hours",
    ],
    "Water Leakage": [
        "water pipe leaking on the roadside",
        "continuous water leakage near the market",
        "pipeline burst and water overflowing",
    ],
    "Sewage": [
        "sewage overflow near houses",
        "dirty sewage water on the street",
        "sewer line is blocked and overflowing",
    ],
    "Drainage": [
        "drainage blocked causing stagnant water",
        "storm drain clogged after rain",
        "water not draining from roadside channel",
    ],
    "Garbage": [
        "garbage pile not collected for days",
        "overflowing trash bin creating bad smell",
        "waste dumped on roadside",
    ],
    "Waste Disposal": [
        "waste disposal point is overflowing",
        "public dustbin area full of waste",
        "solid waste collection missed in our area",
    ],
    "Park": [
        "park maintenance issue with broken benches",
        "public park lights and grass not maintained",
        "children park equipment damaged",
    ],
    "Tree": [
        "tree branch fallen on the road",
        "dangerous tree leaning near electric wires",
        "tree needs trimming in public area",
    ],
    "Safety Hazard": [
        "open manhole creating safety hazard",
        "dangerous exposed wires in public area",
        "unsafe civic condition causing accident risk",
    ],
    "Other": [
        "general civic problem in my locality",
        "public issue needs municipal attention",
        "local complaint regarding civic maintenance",
    ],
}

PRIORITY_KEYWORDS = {
    "critical": 5,
    "urgent": 5,
    "danger": 4,
    "hazard": 4,
    "hospital": 4,
    "school": 3,
    "accident": 4,
    "injury": 4,
    "blocked": 3,
    "overflow": 3,
    "leak": 2,
    "dark": 2,
    "night": 2,
    "traffic": 2,
    "children": 2,
    "elderly": 2,
    "smell": 1,
}

HIGH_PRIORITY_CATEGORIES = {"Sewage", "Safety Hazard", "Power Outage", "Water Leakage"}
MEDIUM_PRIORITY_CATEGORIES = {"Street Light", "Pothole", "Road Damage", "Drainage", "Garbage", "Tree"}
IMAGE_CATEGORY_HINTS = {
    "dark_scene": ("Street Light", 0.76, "Image appears unusually dark, which often matches street light or power issues."),
    "water_scene": ("Water Leakage", 0.82, "Image shows strong blue-gray water-like tones consistent with leakage or drainage issues."),
    "green_scene": ("Tree", 0.72, "Image contains strong green coverage, which often relates to tree or park maintenance."),
    "brown_road_scene": ("Pothole", 0.68, "Image contains rough road-like tones that may indicate pothole or road damage."),
}
RULE_BASED_CATEGORY_PATTERNS = {
    "Pothole": [
        r"\bpothole\b",
        r"\bpot hole\b",
        r"\bpath hole\b",
        r"\bpathhole\b",
        r"\bpotwhole\b",
        r"\broad hole\b",
        r"\bdeep hole\b",
        r"\bcrater\b",
    ],
    "Road Damage": [
        r"\broad damage\b",
        r"\bdamaged road\b",
        r"\bbroken road\b",
        r"\broad crack\b",
        r"\bcracked road\b",
        r"\basphalt\b",
    ],
    "Street Light": [
        r"\bstreet light\b",
        r"\bstreetlight\b",
        r"\blight pole\b",
        r"\blamp post\b",
        r"\bno light\b",
        r"\bdark road\b",
    ],
    "Power Outage": [
        r"\bpower outage\b",
        r"\bcurrent gone\b",
        r"\bno current\b",
        r"\belectricity gone\b",
        r"\bpower cut\b",
    ],
    "Water Leakage": [
        r"\bwater leak\b",
        r"\bwater leakage\b",
        r"\bpipe leak\b",
        r"\bpipeline burst\b",
        r"\bwater overflowing\b",
    ],
    "Sewage": [
        r"\bsewage\b",
        r"\bsewer\b",
        r"\bdirty water\b",
        r"\btoilet water\b",
        r"\bdrain water smell\b",
    ],
    "Drainage": [
        r"\bdrainage\b",
        r"\bblocked drain\b",
        r"\bdrain blocked\b",
        r"\bstagnant water\b",
        r"\bwater logging\b",
    ],
    "Garbage": [
        r"\bgarbage\b",
        r"\btrash\b",
        r"\bwaste pile\b",
        r"\bdustbin overflow\b",
        r"\bbad smell garbage\b",
    ],
    "Tree": [
        r"\btree branch\b",
        r"\bfallen tree\b",
        r"\bleaning tree\b",
        r"\btree trimming\b",
    ],
    "Safety Hazard": [
        r"\bopen manhole\b",
        r"\bexposed wire\b",
        r"\bdangerous\b",
        r"\bhazard\b",
        r"\baccident risk\b",
    ],
}
BASE_DIR = os.path.dirname(__file__)
MODEL_DIR = os.path.join(BASE_DIR, "models")
TRAINING_REPORT_PATH = os.path.join(MODEL_DIR, "training_report.json")
DUPLICATE_DISTANCE_THRESHOLD_METERS = float(os.environ.get("DUPLICATE_DISTANCE_THRESHOLD_METERS", "200"))
DUPLICATE_REVIEW_SCORE_THRESHOLD = float(os.environ.get("DUPLICATE_REVIEW_SCORE_THRESHOLD", "0.45"))
DUPLICATE_BLOCK_SCORE_THRESHOLD = float(os.environ.get("DUPLICATE_BLOCK_SCORE_THRESHOLD", "0.62"))


def normalize_text(value: str | None) -> str:
    text = re.sub(r"\s+", " ", (value or "").strip()).lower()
    typo_map = {
        "pathhole": "pothole",
        "path hole": "pothole",
        "pot hole": "pothole",
        "potwhole": "pothole",
        "streat light": "street light",
        "streetligt": "street light",
        "garabage": "garbage",
        "dranage": "drainage",
        "sevarage": "sewage",
    }
    for typo, corrected in typo_map.items():
        text = text.replace(typo, corrected)
    return text


def title_case_label(value: str) -> str:
    return " ".join(word.capitalize() for word in value.split())


def keyword_examples_from_category(category: str) -> list[str]:
    words = re.findall(r"[a-zA-Z]+", category.lower())
    if not words:
        return []
    base = " ".join(words)
    return [
        f"{base} complaint reported by citizen",
        f"{base} issue needs municipal attention",
        f"{base} problem in public area",
    ]


def build_category_examples(departments: Iterable[dict]) -> dict[str, list[str]]:
    examples = {category: samples[:] for category, samples in DEFAULT_CATEGORY_EXAMPLES.items()}
    for department in departments:
        for raw_category in department.get("categories", []) or []:
            category = title_case_label(str(raw_category).strip())
            if not category:
                continue
            examples.setdefault(category, []).extend(keyword_examples_from_category(category))
    return examples


@dataclass(frozen=True)
class ModelBundle:
    categories: tuple[str, ...]
    vectorizer: object | None
    classifier: object | None
    training_texts: tuple[str, ...]
    training_labels: tuple[str, ...]


@dataclass(frozen=True)
class TrainedArtifacts:
    category_vectorizer: object | None
    category_model: object | None
    priority_vectorizer: object | None
    priority_model: object | None
    metadata: dict


@lru_cache(maxsize=12)
def build_model(signature: tuple[tuple[str, tuple[str, ...]], ...]) -> ModelBundle:
    category_map = {category: list(samples) for category, samples in signature}
    categories = tuple(category_map.keys())
    training_texts: list[str] = []
    training_labels: list[str] = []

    for category, samples in category_map.items():
        for sample in samples:
            training_texts.append(sample)
            training_labels.append(category)

    if TfidfVectorizer is None or LogisticRegression is None:
        return ModelBundle(categories, None, None, tuple(training_texts), tuple(training_labels))

    vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words="english")
    matrix = vectorizer.fit_transform(training_texts)
    classifier = LogisticRegression(max_iter=1000, class_weight="balanced")
    classifier.fit(matrix, training_labels)
    return ModelBundle(categories, vectorizer, classifier, tuple(training_texts), tuple(training_labels))


def signature_from_examples(category_examples: dict[str, list[str]]) -> tuple[tuple[str, tuple[str, ...]], ...]:
    return tuple(sorted((category, tuple(samples)) for category, samples in category_examples.items()))


def rule_based_category_hint(text: str) -> tuple[str | None, float, str]:
    normalized = normalize_text(text)
    if not normalized:
        return None, 0.0, ""

    for category, patterns in RULE_BASED_CATEGORY_PATTERNS.items():
        for pattern in patterns:
            if re.search(pattern, normalized):
                return category, 0.93, f"Matched civic rule for {category.lower()}."
    return None, 0.0, ""


@lru_cache(maxsize=1)
def load_trained_artifacts() -> TrainedArtifacts:
    metadata = {}
    if os.path.exists(TRAINING_REPORT_PATH):
        try:
            with open(TRAINING_REPORT_PATH, "r", encoding="utf-8") as file:
                metadata = json.load(file)
        except Exception:
            metadata = {}

    def load_if_exists(filename: str):
        path = os.path.join(MODEL_DIR, filename)
        if os.path.exists(path):
            try:
                return joblib.load(path)
            except Exception:
                return None
        return None

    return TrainedArtifacts(
        category_vectorizer=load_if_exists("category_vectorizer.joblib"),
        category_model=load_if_exists("category_model.joblib"),
        priority_vectorizer=load_if_exists("priority_vectorizer.joblib"),
        priority_model=load_if_exists("priority_model.joblib"),
        metadata=metadata,
    )


def predict_category(text: str, model: ModelBundle, trained: TrainedArtifacts | None = None) -> tuple[str, float]:
    normalized = normalize_text(text)
    if not normalized:
        return "Other", 0.0

    if trained and trained.category_vectorizer is not None and trained.category_model is not None:
        try:
            matrix = trained.category_vectorizer.transform([normalized])
            prediction = trained.category_model.predict(matrix)[0]
            if hasattr(trained.category_model, "predict_proba"):
                confidence = max(trained.category_model.predict_proba(matrix)[0])
            else:
                confidence = 0.8
            return str(prediction), float(confidence)
        except Exception:
            pass

    if model.vectorizer is not None and model.classifier is not None:
        matrix = model.vectorizer.transform([normalized])
        prediction = model.classifier.predict(matrix)[0]
        confidence = max(model.classifier.predict_proba(matrix)[0])
        return prediction, float(confidence)

    # Fallback when sklearn is unavailable.
    best_category = "Other"
    best_score = 0
    words = set(normalized.split())
    for category in model.categories:
        category_words = set(normalize_text(category).split())
        score = len(words & category_words)
        if score > best_score:
            best_category = category
            best_score = score
    confidence = min(0.95, 0.3 + best_score * 0.2) if best_score else 0.25
    return best_category, confidence


def find_department_for_category(departments: Iterable[dict], category: str) -> tuple[str | None, str]:
    normalized_category = normalize_text(category)
    for department in departments:
        categories = [normalize_text(item) for item in department.get("categories", []) or []]
        if normalized_category in categories:
            return str(department["_id"]), department["name"]

    for department in departments:
        if normalize_text(department.get("name")) in {"public safety", "municipal services"}:
            return str(department["_id"]), department["name"]

    return None, "Unassigned"


def simple_text_similarity(text_a: str, text_b: str) -> float:
    tokens_a = set(normalize_text(text_a).split())
    tokens_b = set(normalize_text(text_b).split())
    if not tokens_a or not tokens_b:
        return 0.0
    overlap = len(tokens_a & tokens_b)
    return overlap / max(1, len(tokens_a | tokens_b))


def exact_or_prefix_title_similarity(title_a: str, title_b: str) -> float:
    normalized_a = normalize_text(title_a)
    normalized_b = normalize_text(title_b)
    if not normalized_a or not normalized_b:
        return 0.0
    if normalized_a == normalized_b:
        return 1.0
    if normalized_a in normalized_b or normalized_b in normalized_a:
        return 0.92
    return simple_text_similarity(normalized_a, normalized_b)


def semantic_similarity(texts: list[str]) -> list[list[float]]:
    if len(texts) < 2:
        return [[1.0]]

    if TfidfVectorizer is not None and cosine_similarity is not None:
        vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words="english")
        matrix = vectorizer.fit_transform(texts)
        values = cosine_similarity(matrix)
        return values.tolist()

    size = len(texts)
    values = [[0.0] * size for _ in range(size)]
    for i in range(size):
        values[i][i] = 1.0
        for j in range(i + 1, size):
            score = simple_text_similarity(texts[i], texts[j])
            values[i][j] = score
            values[j][i] = score
    return values


def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    radius = 6371000
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return 2 * radius * math.asin(math.sqrt(a))


def is_duplicate_distance(distance_meters: float | None) -> bool:
    return distance_meters is not None and distance_meters <= DUPLICATE_DISTANCE_THRESHOLD_METERS


def is_duplicate_score(score: float | None) -> bool:
    return score is not None and score >= DUPLICATE_BLOCK_SCORE_THRESHOLD


def calculate_duplicate_signal(
    db,
    title: str,
    description: str,
    lat: float | None,
    lon: float | None,
    category: str | None = None,
    department_name: str | None = None,
) -> dict:
    current_text = normalize_text(f"{title} {description}")
    normalized_category = normalize_text(category)
    normalized_department = normalize_text(department_name)
    if not current_text:
        return {
            "score": 0.0,
            "duplicate_issue_id": None,
            "similarity": 0.0,
            "distance_meters": None,
            "message": "Not enough information to check similar complaints.",
        }

    issues = list(
        db.issues.find(
            {},
            {
                "title": 1,
                "description": 1,
                "latitude": 1,
                "longitude": 1,
                "status": 1,
                "category": 1,
                "department_name": 1,
            },
        )
        .sort("created_at", -1)
        .limit(150)
    )
    if not issues:
        return {
            "score": 0.0,
            "duplicate_issue_id": None,
            "similarity": 0.0,
            "distance_meters": None,
            "message": "No previous complaints available for duplicate analysis.",
        }

    texts = [current_text] + [normalize_text(f"{item.get('title', '')} {item.get('description', '')}") for item in issues]
    similarity_matrix = semantic_similarity(texts)

    best = {
        "score": 0.0,
        "duplicate_issue_id": None,
        "similarity": 0.0,
        "distance_meters": None,
        "message": "No likely duplicate found.",
    }

    for index, issue in enumerate(issues, start=1):
        semantic_score = float(similarity_matrix[0][index])
        lexical_score = simple_text_similarity(current_text, normalize_text(f"{issue.get('title', '')} {issue.get('description', '')}"))
        title_score = exact_or_prefix_title_similarity(title, issue.get("title", ""))
        similarity = max(semantic_score, lexical_score, title_score)
        distance_score = 0.0
        category_score = 0.0
        department_score = 0.0
        distance_meters = None
        existing_lat = issue.get("latitude")
        existing_lon = issue.get("longitude")
        existing_category = normalize_text(issue.get("category"))
        existing_department = normalize_text(issue.get("department_name"))

        if normalized_category and existing_category:
            category_score = 1.0 if normalized_category == existing_category else 0.0

        if normalized_department and existing_department:
            department_score = 1.0 if normalized_department == existing_department else 0.0

        if lat is not None and lon is not None and existing_lat is not None and existing_lon is not None:
            distance_meters = haversine_distance(lat, lon, existing_lat, existing_lon)
            if distance_meters <= DUPLICATE_DISTANCE_THRESHOLD_METERS:
                distance_score = 1.0
            elif distance_meters <= 500:
                distance_score = 0.6
            elif distance_meters <= 1000:
                distance_score = 0.25

        score = round(
            (similarity * 0.55)
            + (distance_score * 0.25)
            + (category_score * 0.12)
            + (department_score * 0.08),
            3,
        )
        if score > best["score"]:
            if is_duplicate_distance(distance_meters) and is_duplicate_score(score):
                message = f"Potential duplicate complaint found within {DUPLICATE_DISTANCE_THRESHOLD_METERS} meters."
            elif is_duplicate_distance(distance_meters):
                message = (
                    f"A nearby issue was found within {DUPLICATE_DISTANCE_THRESHOLD_METERS} meters, "
                    "but the text match confidence is moderate."
                )
            elif distance_meters is not None:
                message = f"Similar complaint found, but it is more than {DUPLICATE_DISTANCE_THRESHOLD_METERS} meters away."
            else:
                message = "Similar complaint found, but location could not confirm a nearby duplicate."
            best = {
                "score": score,
                "duplicate_issue_id": str(issue["_id"]),
                "similarity": round(similarity, 3),
                "distance_meters": round(distance_meters, 1) if distance_meters is not None else None,
                "message": message if score >= DUPLICATE_REVIEW_SCORE_THRESHOLD else "No likely duplicate found.",
            }

    return best


def predict_priority(
    category: str,
    title: str,
    description: str,
    duplicate_score: float,
    trained: TrainedArtifacts | None = None,
) -> tuple[str, float]:
    text = normalize_text(f"{title} {description}")
    if trained and trained.priority_vectorizer is not None and trained.priority_model is not None:
        try:
            feature_text = normalize_text(f"{category} {title} {description}")
            matrix = trained.priority_vectorizer.transform([feature_text])
            prediction = trained.priority_model.predict(matrix)[0]
            if hasattr(trained.priority_model, "predict_proba"):
                confidence = max(trained.priority_model.predict_proba(matrix)[0])
            else:
                confidence = 0.8
            return str(prediction), float(confidence)
        except Exception:
            pass

    score = 0

    if category in HIGH_PRIORITY_CATEGORIES:
        score += 4
    elif category in MEDIUM_PRIORITY_CATEGORIES:
        score += 2

    for keyword, weight in PRIORITY_KEYWORDS.items():
        if keyword in text:
            score += weight

    if duplicate_score >= 0.75:
        score += 2
    elif duplicate_score >= 0.45:
        score += 1

    if score >= 8:
        return "High", min(0.99, 0.65 + score / 20)
    if score >= 4:
        return "Medium", min(0.95, 0.55 + score / 20)
    return "Low", min(0.9, 0.45 + score / 20)


def analyze_image_signals(image_path: str | None) -> dict:
    if not image_path:
        return {
            "category": None,
            "confidence": 0.0,
            "signal": None,
            "insight": "",
            "brightness": None,
            "contrast": None,
        }

    try:
        with Image.open(image_path) as image:
            rgb_image = image.convert("RGB").resize((128, 128))
            stat = ImageStat.Stat(rgb_image)
            mean_r, mean_g, mean_b = stat.mean[:3]
            std_r, std_g, std_b = stat.stddev[:3]
            brightness = (mean_r + mean_g + mean_b) / 3
            contrast = (std_r + std_g + std_b) / 3

            if brightness < 60:
                category, confidence, insight = IMAGE_CATEGORY_HINTS["dark_scene"]
                signal = "dark_scene"
            elif mean_b > mean_r + 12 and mean_b > mean_g + 8:
                category, confidence, insight = IMAGE_CATEGORY_HINTS["water_scene"]
                signal = "water_scene"
            elif mean_g > mean_r + 10 and mean_g > mean_b + 10:
                category, confidence, insight = IMAGE_CATEGORY_HINTS["green_scene"]
                signal = "green_scene"
            elif contrast > 42 and mean_r > 85 and mean_g > 80 and mean_b < 115:
                category, confidence, insight = IMAGE_CATEGORY_HINTS["brown_road_scene"]
                signal = "brown_road_scene"
            else:
                signal = "generic_scene"
                category = None
                confidence = 0.0
                insight = "Image uploaded successfully, but visual cues were not strong enough to change the AI classification."

            return {
                "category": category,
                "confidence": round(confidence, 3),
                "signal": signal,
                "insight": insight,
                "brightness": round(brightness, 1),
                "contrast": round(contrast, 1),
            }
    except Exception:
        return {
            "category": None,
            "confidence": 0.0,
            "signal": "unreadable",
            "insight": "The uploaded image could not be analyzed, so AI used text-only classification.",
            "brightness": None,
            "contrast": None,
        }


def build_summary(category: str, priority: str, department_name: str, title: str, description: str) -> str:
    short_desc = re.sub(r"\s+", " ", (description or "").strip())
    if len(short_desc) > 140:
        short_desc = short_desc[:137].rstrip() + "..."
    if not short_desc:
        short_desc = "Citizen submitted a civic complaint."
    return f"{priority} priority {category.lower()} complaint for {department_name}: {title.strip()}. {short_desc}"


def analyze_issue(
    db,
    title: str,
    description: str,
    latitude: float | None = None,
    longitude: float | None = None,
    image_path: str | None = None,
) -> dict:
    departments = list(db.departments.find({}, {"name": 1, "categories": 1}))
    category_examples = build_category_examples(departments)
    model = build_model(signature_from_examples(category_examples))
    trained = load_trained_artifacts()

    combined_text = f"{title}. {description}".strip()
    predicted_category, category_confidence = predict_category(combined_text, model, trained)
    rule_category, rule_confidence, rule_note = rule_based_category_hint(combined_text)
    if rule_category and rule_confidence >= category_confidence:
        predicted_category = rule_category
        category_confidence = rule_confidence

    image_analysis = analyze_image_signals(image_path)
    if image_analysis["category"] and (
        image_analysis["confidence"] > category_confidence or category_confidence < 0.6
    ):
        predicted_category = image_analysis["category"]
        category_confidence = max(category_confidence, image_analysis["confidence"])

    ai_department_id, ai_department_name = find_department_for_category(departments, predicted_category)
    duplicate_signal = calculate_duplicate_signal(
        db,
        title,
        description,
        latitude,
        longitude,
        predicted_category,
        ai_department_name,
    )
    priority, priority_confidence = predict_priority(
        predicted_category,
        title,
        description,
        duplicate_signal["score"],
        trained,
    )
    if image_analysis["signal"] == "dark_scene" and priority == "Low":
        priority, priority_confidence = "Medium", max(priority_confidence, 0.65)
    if image_analysis["signal"] in {"water_scene", "brown_road_scene"} and priority == "Medium":
        priority, priority_confidence = "High", max(priority_confidence, 0.72)
    summary = build_summary(predicted_category, priority, ai_department_name, title, description)

    return {
        "category": predicted_category,
        "category_confidence": round(category_confidence, 3),
        "department_id": ai_department_id,
        "department_name": ai_department_name,
        "priority": priority,
        "priority_confidence": round(priority_confidence, 3),
        "summary": summary,
        "duplicate_score": duplicate_signal["score"],
        "duplicate_issue_id": duplicate_signal["duplicate_issue_id"],
        "duplicate_similarity": duplicate_signal["similarity"],
        "duplicate_distance_meters": duplicate_signal["distance_meters"],
        "duplicate_message": duplicate_signal["message"],
        "image_signal": image_analysis["signal"],
        "image_insight": image_analysis["insight"],
        "image_category": image_analysis["category"],
        "image_category_confidence": image_analysis["confidence"],
        "image_brightness": image_analysis["brightness"],
        "image_contrast": image_analysis["contrast"],
        "model_source": "trained" if trained.category_model is not None else "bootstrap",
        "training_report": trained.metadata,
        "rule_based_match": rule_note,
    }
