# app.py – Streamlit App for Disease Classifier

import streamlit as st
import joblib

# Load model and vectorizer
model = joblib.load("disease_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# UI
st.set_page_config(page_title="Symptom Disease Classifier")
st.title("🩺 Disease Classifier from Symptoms")
st.write("Enter symptoms separated by spaces (e.g., `fever cough fatigue`)")

user_input = st.text_input("Symptoms")

if st.button("Predict Disease"):
    if user_input.strip() == "":
        st.warning("Please enter some symptoms.")
    else:
        input_vec = vectorizer.transform([user_input])
        prediction = model.predict(input_vec)[0]
        st.success(f"🧠 Predicted Disease: **{prediction}**")
