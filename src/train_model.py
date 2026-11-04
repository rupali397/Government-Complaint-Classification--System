
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import matplotlib.pyplot as plt

from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    cross_val_score
)
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

# 1. Dataset file paths
PROJECT_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = PROJECT_DIR / "models"

ORIGINAL_PATH = PROJECT_DIR / "metadata.csv"
ADDITIONAL_PATH = PROJECT_DIR / "additional_complaints.csv"


# 2. Function to clean dataset
def clean_dataset(data):
    data = data.copy()

    data.columns = data.columns.map(
        lambda col: str(col).strip().strip('"').strip("'")
    )

    required_columns = {"transcript", "category"}

    if not required_columns.issubset(data.columns):
        raise ValueError(
            f"Required columns missing. Found: {data.columns.tolist()}"
        )

    data = data.dropna(
        subset=["transcript", "category"]
    ).copy()

    data["transcript"] = data["transcript"].astype(str).str.strip()
    data["category"] = data["category"].astype(str).str.strip()

    # Remove numbering and surrounding quotation marks
    data["transcript"] = data["transcript"].str.replace(
    r"^\s*\d+\.\s*",
    "",
    regex=True
    )

    data["transcript"] = data["transcript"].str.strip(" \"'“”‘’")

    data = data[
        (data["transcript"] != "") &
        (data["category"] != "")
    ].copy()

    return data


# 3. Load original and additional datasets
original_df = pd.read_csv(ORIGINAL_PATH)
additional_df = pd.read_csv(ADDITIONAL_PATH)

original_df = clean_dataset(original_df)
additional_df = clean_dataset(additional_df)

print("\n--- RATION EXAMPLES WITH FULL TEXT ---")

ration_data = additional_df[
    additional_df["category"] == "ration_aadhar"
]

for i, complaint in enumerate(ration_data["transcript"], start=1):
    print(f"{i}. {complaint}")

print("\n--- SANITATION EXAMPLES ---")
print(additional_df[
    additional_df["category"] == "sanitation"
]["transcript"].head(10).to_string(index=False))

# Keep only categories present in the original dataset
valid_categories = original_df["category"].unique()

original_df = original_df[
    original_df["category"].isin(valid_categories)
].copy()

additional_df = additional_df[
    additional_df["category"].isin(valid_categories)
].copy()

# Separate original data
X_original = original_df["transcript"]
y_original = original_df["category"]

# 5. Add additional complaints ONLY to training data

# Split original data into training and final testing
X_train, X_test, y_train, y_test = train_test_split(
    X_original,
    y_original,
    test_size=0.25,
    random_state=42,
    stratify=y_original
)

# Split training data further into model training and validation
X_train, X_val, y_train, y_val = train_test_split(
    X_train,
    y_train,
    test_size=0.20,
    random_state=42,
    stratify=y_train
)

# Add additional complaints ONLY to training data
X_train = pd.concat(
    [X_train, additional_df["transcript"]],
    ignore_index=True
)

y_train = pd.concat(
    [y_train, additional_df["category"]],
    ignore_index=True
)

# Remove ambiguous complaints from training data

ambiguous_complaints = [
    "कई बार शिकायत की, लेकिन कोई कार्रवाई नहीं होती।",
    "स्टाफ असहयोगी और लापरवाह है।",
    "Complaints are raised but no action is taken.",
    "Staff is uncooperative and careless.",
    "We had to wait for hours, and no one responded.",
    "हमें घंटों इंतजार करना पड़ा, लेकिन कोई जवाब नहीं मिला।",
    "The helpline number is always busy or unreachable.",
    "कई बार शिकायत की, लेकिन कोई समाधान नहीं हुआ।"
]

# Reset indexes before filtering
X_train = X_train.reset_index(drop=True)
y_train = y_train.reset_index(drop=True)

# Keep only clear complaints
keep_mask = ~X_train.isin(ambiguous_complaints)

X_train = X_train[keep_mask].reset_index(drop=True)
y_train = y_train[keep_mask].reset_index(drop=True)

print("\nAmbiguous complaints removed:", len(ambiguous_complaints))
print("Updated training complaints:", len(X_train))


# DATASET QUALITY AUDIT

print("\n--- TRAINING DATA CATEGORY AUDIT ---")

for category in sorted(y_train.unique()):
    print(f"\n\nCATEGORY: {category}")

    category_data = X_train[y_train == category]

    print(f"Total complaints: {len(category_data)}")

    for i, complaint in enumerate(category_data, start=1):
        print(f"{i}. {complaint}")


# 6. Dataset summary
print("\n--- FINAL DATASET SUMMARY ---")
print("Original complaints:", len(original_df))
print("Additional complaints:", len(additional_df))
print("Training complaints:", len(X_train))
print("Testing complaints:", len(X_test))

print("\nTraining category counts:")
print(y_train.value_counts())

print("\nTesting category counts:")
print(y_test.value_counts())


# 4. Combine word patterns and character patterns
text_features = FeatureUnion([
    (
        "word",
        TfidfVectorizer(
            ngram_range=(1, 2),
            sublinear_tf=True,
            strip_accents="unicode"
        )
    ),
    (
        "character",
        TfidfVectorizer(
            analyzer="char",
            ngram_range=(3, 5),
            sublinear_tf=True,
            strip_accents="unicode"
        )
    )
])

# 5. Compare three ML models
models = {
    "Logistic Regression": LogisticRegression(
        max_iter=2000,
        class_weight="balanced"
    ),
    "Linear SVM": LinearSVC(
        class_weight="balanced",
        random_state=42
    ),
    "Naive Bayes": MultinomialNB()
}

cv = StratifiedKFold(
    n_splits=3,
    shuffle=True,
    random_state=42
)

results = {}

print("\n--- MODEL COMPARISON (TRAINING DATA) ---")

for name, classifier in models.items():
    pipeline = Pipeline([
        ("features", text_features),
        ("classifier", classifier)
    ])

    scores = cross_val_score(
        pipeline,
        X_train,
        y_train,
        cv=cv,
        scoring="accuracy"
    )

    results[name] = scores.mean()

    print(
        f"{name}: "
        f"CV accuracy = {scores.mean() * 100:.2f}% "
        f"(fold scores: {[round(s * 100, 2) for s in scores]})"
    )

# 6. Select the model with the best average CV score
best_name = "Logistic Regression" 
print("\nSelected model:", best_name)

best_model = Pipeline([
    ("features", text_features),
    ("classifier", models[best_name])
])

# 7. Train selected model and evaluate on held-out test data
best_model.fit(X_train, y_train)
y_pred = best_model.predict(X_test)

# Confidence-based evaluation on validation data
val_pred = best_model.predict(X_val)

probabilities = best_model.predict_proba(X_val)

confidence_scores = probabilities.max(axis=1)
sorted_probabilities = -np.sort(-probabilities, axis=1)
margins = sorted_probabilities[:, 0] - sorted_probabilities[:, 1]

# Confidence analysis on held-out test data
test_probabilities = best_model.predict_proba(X_test)

test_confidence = test_probabilities.max(axis=1)

sorted_test_probabilities = -np.sort(
    -test_probabilities,
    axis=1
)

test_margins = (
    sorted_test_probabilities[:, 0]
    - sorted_test_probabilities[:, 1]
)

print("\n--- TEST CONFIDENCE ANALYSIS ---")

print("\n--- VALIDATION CONFIDENCE EVALUATION ---")

print("\n--- THRESHOLD COMPARISON ---")

for threshold, margin_threshold in [
    (0.20, 0.02),
    (0.22, 0.03),
    (0.25, 0.05),
    (0.28, 0.08),
    (0.30, 0.10)
]:
    accepted = (
        (confidence_scores >= threshold) &
        (margins >= margin_threshold)
    )

    total_accepted = accepted.sum()

    if total_accepted > 0:
        accepted_accuracy = accuracy_score(
            y_val.iloc[accepted],
            val_pred[accepted]
        ) * 100
    else:
        accepted_accuracy = 0

    coverage = accepted.mean() * 100

    print(
        f"Confidence: {threshold:.2f} | "
        f"Margin: {margin_threshold:.2f} | "
        f"Accepted: {total_accepted}/18 | "
        f"Coverage: {coverage:.2f}% | "
        f"Accepted Accuracy: {accepted_accuracy:.2f}%"
    )

print("\n--- VALIDATION CONFIDENCE DETAILS ---")

for i in range(len(X_val)):
    print(
        f"Actual: {y_val.iloc[i]} | "
        f"Predicted: {val_pred[i]} | "
        f"Confidence: {confidence_scores[i]:.3f} | "
        f"Margin: {margins[i]:.3f}"
    )

results = pd.DataFrame({
    "Actual": y_test.values,
    "Predicted": y_pred,
    "Confidence": test_confidence,
    "Margin": test_margins,
    "Complaint": X_test.values
})

wrong_predictions = results[
    results["Actual"] != results["Predicted"]
]

print("\n--- INCORRECTLY PREDICTED COMPLAINTS ---")
print(wrong_predictions.to_string(index=False))

print("\nTotal incorrect predictions:", len(wrong_predictions))

print("\n--- INCORRECT PREDICTIONS ---")
print(wrong_predictions.to_string(index=False))

test_accuracy = accuracy_score(y_test, y_pred)
print(f"\nFinal Test Accuracy: {test_accuracy * 100:.2f}%")


print("\n--- FINAL TEST RESULTS ---")
print("Training complaints:", len(X_train))
print("Testing complaints:", len(X_test))
print(f"Test accuracy: {test_accuracy * 100:.2f}%")

print("\n--- CLASSIFICATION REPORT ---")
print(
    classification_report(
        y_test,
        y_pred,
        zero_division=0
    )
)

# 8. Confusion matrix
labels = best_model.named_steps["classifier"].classes_

cm = confusion_matrix(
    y_test,
    y_pred,
    labels=labels
)

print("\n--- CONFUSION MATRIX ---")
print("Category order:", labels)
print(cm)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=labels
)


# Show incorrectly predicted complaints

results = pd.DataFrame({
    "Actual": y_test.values,
    "Predicted": y_pred,
    "Complaint": X_test.values
})

wrong_predictions = results[
    results["Actual"] != results["Predicted"]
]

print("\n--- TRAINING EXAMPLES BY CATEGORY ---")

for category in sorted(y_train.unique()):
    print(f"\nCATEGORY: {category}")

    examples = X_train[y_train == category].sample(
    n=min(10, (y_train == category).sum()),
    random_state=42
    )

    for example in examples:
        print("-", example)

print("\n--- INCORRECTLY PREDICTED COMPLAINTS ---")

print(wrong_predictions.to_string(index=False))

print("\nTotal incorrect predictions:", len(wrong_predictions))


disp.plot(xticks_rotation=45)
plt.tight_layout()
plt.show()

# 9. Save the selected model
MODEL_DIR.mkdir(parents=True, exist_ok=True)

model_path = MODEL_DIR / "complaint_classifier.joblib"
joblib.dump(best_model, model_path)

print("CHECKPOINT 1: Model saved")

# Prepare evaluation results
evaluation_results = {
    "test_accuracy": test_accuracy,
    "classification_report": classification_report(
        y_test,
        y_pred,
        output_dict=True,
        zero_division=0
    ),
    "confusion_matrix": cm,
    "category_labels": labels.tolist()
}

# Save evaluation results
evaluation_path = MODEL_DIR / "evaluation_results.joblib"

print("Evaluation path:", evaluation_path)
print("Evaluation data keys:", evaluation_results.keys())

joblib.dump(evaluation_results, evaluation_path)

print("CHECKPOINT 2: Evaluation saved")
print("Evaluation results saved at:", evaluation_path)

print("Model saved at:", model_path)
print("Training completed!")
