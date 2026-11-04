# Government Complaint Classification System

An ML-based application that automatically classifies government complaints into different categories using Natural Language Processing (NLP) and Machine Learning.

## 1. Project Overview

The Government Complaint Classification System helps categorize citizen complaints automatically. Users can enter an individual complaint or upload a CSV file containing multiple complaints to get predictions.

## 2. Problem Statement

Government departments receive complaints related to different public services. Manually categorizing these complaints can be time-consuming.

This project uses Machine Learning to predict the appropriate complaint category from the complaint text.

## 3. Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- TF-IDF Vectorization
- Logistic Regression
- Streamlit
- Matplotlib
- Joblib

## 4. Complaint Categories

The model supports six categories:

- Electricity
- Healthcare
- Ration and Aadhaar
- Roads and Transport
- Sanitation
- Water Supply

The dataset contains English and Hindi complaints.

## 5. Machine Learning Workflow

1. Load and prepare the complaint dataset.
2. Clean and process complaint text.
3. Extract text features using Word TF-IDF and Character TF-IDF.
4. Train and evaluate multiple ML classifiers.
5. Select Logistic Regression as the final model.
6. Save the trained model using Joblib.
7. Use Streamlit to make predictions.

## 6. Model Performance

| Metric | Result |
|---|---:|
| Training samples | 188 |
| Testing samples | 30 |
| Test Accuracy | 70.00% |
| Correct Predictions | 21 / 30 |

**Note:** The reported accuracy is based on a test set of 30 complaints. Performance on real-world complaints may differ.

## 7. Application Features

- Single complaint classification
- Bulk complaint classification through CSV upload
- Downloadable prediction results
- Model performance dashboard
- Classification report
- Confusion matrix heatmap
- Insufficient information detection

## 8. Project Structure

```text
Government_Complaints_Dataset/
│
├── app.py
├── src/
│   └── train_model.py
│
├── models/
│   ├── complaint_classifier.joblib
│   ├── complaint_classifier_backup.joblib
│   └── evaluation_results.joblib
│
└── README.md
```

## 9. Installation and Setup

Install the required packages:

```bash
pip install pandas numpy scikit-learn streamlit matplotlib joblib
```

Run the application from the main project folder:

```bash
python -m streamlit run app.py
```

The application will open in your web browser.

## 10. Future Improvements

- Increase dataset size and diversity.
- Improve Hindi and Hinglish complaint classification.
- Explore advanced NLP and transformer-based models.
- Improve model accuracy through systematic experimentation.
- Test performance on unseen real-world complaints.

## 11. Project Status

**Status:** Working ML application with single and bulk prediction features and a performance dashboard.

---

*Developed as a Machine Learning and NLP learning project.*