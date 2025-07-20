# train.py – Disease Classifier from Symptoms (TF-IDF + Naive Bayes)

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score
import joblib

# Synthetic dataset (or replace with real one)
data = {
    "symptoms": [
        "fever cough fatigue",
        "headache blurred vision nausea",
        "chest pain shortness of breath cough",
        "rash itching redness",
        "frequent urination thirst blurred vision",
        "joint pain stiffness fatigue",
        "sore throat cough runny nose",
        "abdominal pain diarrhea vomiting",
    ],
    "disease": [
        "Flu",
        "Migraine",
        "Pneumonia",
        "Allergy",
        "Diabetes",
        "Arthritis",
        "Cold",
        "Gastroenteritis",
    ]
}

# Load data
symptom_df = pd.DataFrame(data)
X = symptom_df['symptoms']
y = symptom_df['disease']

# TF-IDF + Train-test split
vectorizer = TfidfVectorizer()
X_vec = vectorizer.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_vec, y, test_size=0.2, random_state=42)

# Train model
model = MultinomialNB()
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Save model and vectorizer
joblib.dump(model, "disease_model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")
print("✅ Model and vectorizer saved.")
