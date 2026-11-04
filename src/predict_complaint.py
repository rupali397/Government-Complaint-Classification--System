import joblib 
from pathlib import Path
PROJECT_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_DIR / "models" / "complaint_classifier.joblib"
model = joblib.load(MODEL_PATH)
print("=== ML Complaint Classifier ===")
complaint = input("Enter your complaint : ")
if complaint.strip():
    prediction = model.predict([complaint])[0]
    print("Predicted Category:",prediction)
else:
    print("Please enter a valid complaint.")