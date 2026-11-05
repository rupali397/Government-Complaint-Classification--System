import streamlit as st
import joblib
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
# Project folder ka path
PROJECT_DIR = Path(__file__).resolve().parent
# Trained model load krna
MODEL_PATH = PROJECT_DIR / "models" / "complaint_classifier.joblib"
model = joblib.load(MODEL_PATH)

# Load model evaluation results
EVALUATION_PATH = PROJECT_DIR / "models" / "evaluation_results.joblib"

evaluation = joblib.load(EVALUATION_PATH)

# Model Performance Dashboard

st.title("Model Performance Dashboard")
st.subheader("Model Information")

col1, col2 = st.columns(2)

with col1:
    st.info("Algorithm: Logistic Regression")

with col2:
    st.info("Feature Extraction: Word TF-IDF + Character TF-IDF")
    

st.subheader("Overall Model Accuracy")

accuracy = evaluation["test_accuracy"] * 100

st.metric(
    label="Test Accuracy",
    value=f"{accuracy:.2f}%"
)

st.subheader("Dataset Information")

col1, col2 = st.columns(2)

col1.metric("Training Complaints", 188)
col2.metric("Testing Complaints", 30)

st.subheader("Classification Report")

report = pd.DataFrame(
    evaluation["classification_report"]
).transpose()

st.dataframe(report.round(2))

st.subheader("Confusion Matrix")

cm = evaluation["confusion_matrix"]
labels = evaluation["category_labels"]

confusion_df = pd.DataFrame(
    cm,
    index=labels,
    columns=labels
)

fig, ax = plt.subplots(figsize=(10, 6))

image = ax.imshow(cm, cmap="Blues")

ax.set_xticks(range(len(labels)))
ax.set_yticks(range(len(labels)))

ax.set_xticklabels(labels, rotation=45, ha="right")
ax.set_yticklabels(labels)

ax.set_xlabel("Predicted Category")
ax.set_ylabel("Actual Category")
ax.set_title("Confusion Matrix")

for i in range(len(labels)):
    for j in range(len(labels)):
        ax.text(j, i, cm[i, j], ha="center", va="center")

fig.colorbar(image, ax=ax)
fig.tight_layout()

st.pyplot(fig)
plt.close(fig)



# Web app ka title 
st.title("ML Complaint Classifier")
st.write("Enter a complaint to predict its category.")

# Single complaint prediction
complaint = st.text_area("Enter your complaint")

if st.button("Predict Category"):
    if complaint.strip():
        probabilities = model.predict_proba([complaint])[0]

        best_index = probabilities.argmax()
        prediction = model.classes_[best_index]
        confidence = probabilities[best_index]

        sorted_probabilities = sorted(probabilities, reverse=True)
        margin = sorted_probabilities[0] - sorted_probabilities[1]

        if confidence < 0.15:
            st.warning(
                "Insufficient Information: Please provide more details."
            )
        else:
            st.success(f"Predicted Category: {prediction}")
            st.info(f"Confidence: {confidence:.2%}")

    else:
        st.warning("Please enter a complaint.")

# CSV bulk prediction
st.subheader("Bulk Complaint Prediction")

upload_file = st.file_uploader("Upload a CSV file", type=["csv"])

if upload_file is not None:

    try:
        df = pd.read_csv(upload_file)

        if df.empty:
            st.error("The uploaded CSV file is empty.")

        elif "complaint" not in df.columns:
            st.error("CSV must contain a 'complaint' column.")

        else:
            st.success("CSV uploaded successfully!")

            df["complaint"] = df["complaint"].fillna("").astype(str)

            df["predicted_category"] = ""
            df["confidence"] = ""
            df["margin"] = ""
            df["status"] = ""

            mask = df["complaint"].str.strip() != ""

            if mask.any():

                complaints = df.loc[mask, "complaint"]
                probabilities = model.predict_proba(complaints)

                predicted_categories = []
                confidence_scores = []
                statuses = []
                margins = []

                for probs in probabilities:

                    best_index = probs.argmax()
                    prediction = model.classes_[best_index]
                    confidence = probs[best_index]

                    sorted_probs = sorted(probs, reverse=True)
                    margin = sorted_probs[0] - sorted_probs[1]

                    margins.append(f"{margin:.2%}")

                    if confidence < 0.25 or margin < 0.05:
                        predicted_categories.append(
                            "Insufficient Information"
                        )
                        confidence_scores.append(f"{confidence:.2%}")
                        statuses.append("Needs more details")

                    else:
                        predicted_categories.append(prediction)
                        confidence_scores.append(f"{confidence:.2%}")
                        statuses.append("Classified")

                df.loc[mask, "predicted_category"] = predicted_categories
                df.loc[mask, "confidence"] = confidence_scores
                df.loc[mask, "status"] = statuses
                df.loc[mask, "margin"] = margins

            else:
                st.warning("No valid complaints found in the CSV.")

            st.write("Predicted Results")
            st.dataframe(df)

            csv_data = df.to_csv(index=False).encode("utf-8")

            st.download_button(
                "Download Prediction CSV",
                data=csv_data,
                file_name="predicted_complaints.csv",
                mime="text/csv"
            )

    except pd.errors.EmptyDataError:
        st.error("The uploaded CSV file contains no data.")

    except pd.errors.ParserError:
        st.error("Invalid CSV format. Please upload a valid CSV file.")

    except UnicodeDecodeError:
        st.error("Unable to read this file. Please upload a UTF-8 encoded CSV.")

    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")

