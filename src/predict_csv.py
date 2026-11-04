import pandas as pd
import joblib
from pathlib import Path
# Project Folder ka path
PROJECT_DIR = Path(__file__).resolve().parent.parent
# Trained model load krna
MODEL_PATH = PROJECT_DIR / "models" / "complaint_classifier.joblib"
model = joblib.load(MODEL_PATH)
# Input CSV file ka path
INPUT_PATH = PROJECT_DIR / "complaints.csv"
#CSV file read krna
df = pd.read_csv(INPUT_PATH)
# Check karna ki complaint column available hai
if "complaint" not in df.columns:
    print("Error: CSV mein 'complaint' naam ka column hona chahiye.")
else:
    # Empty complaints ko blank text se replace krna 
    df["complaint"] = df["complaint"].fillna("").astype(str)
    # Har complaint ki category predict karna
    df["predicted_category"] = ""
    mask = df["complaint"].str.strip() != ""
    df.loc[mask,"predicted_category"] = model.predict(df.loc[mask,"complaint"])
    # Result ko nayi CSV file mein save krna 
    OUTPUT_PATH = PROJECT_DIR / "predicted_complaints.csv" 
    df.to_csv(OUTPUT_PATH, index = False)
    print("Prediction completed!")
    print("Total complaints:",len(df))
    print("Results saved at :",OUTPUT_PATH)
    print(df.head()) 
