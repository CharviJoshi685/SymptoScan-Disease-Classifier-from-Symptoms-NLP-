# 🩺 Disease Classifier from Symptoms

This project is an intelligent web app that predicts the most likely disease based on user-input symptoms. It uses **Natural Language Processing (NLP)** with **TF-IDF** and **Naive Bayes** for multiclass classification.

---

## 🚀 Features

- Input multiple symptoms as text
- Predict most probable disease
- Simple and fast web interface using **Streamlit**
- Model trained on synthetic symptom–disease data (can be extended)

---

## 🧠 Machine Learning Pipeline

- **Text Vectorization:** TF-IDF
- **Model:** Multinomial Naive Bayes
- **Evaluation Metrics:** Accuracy, Classification Report

---

## 🗂️ Dataset

This project uses a **synthetic dataset** of common diseases and their symptoms. To expand, you can replace it with real-world datasets or medical records.

---

## 📁 File Structure

```
├── app.py               # Streamlit App
├── train.py             # Model training script
├── disease_model.pkl    # Trained model
├── vectorizer.pkl       # TF-IDF vectorizer
├── README.md
├── requirements.txt
├── .gitignore
```

---

## ⚙️ Installation & Usage

### 1. Clone the Repo

```bash
git clone https://github.com/yourusername/symptom-disease-classifier.git
cd symptom-disease-classifier
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Train the Model (optional)

```bash
python train.py
```

### 4. Run the App

```bash
streamlit run app.py
```

---

## 🧪 Sample Input

```
fever cough fatigue
```

## ✅ Sample Output

```
Predicted Disease: Flu
```

---

## 📄 License

This project is open-source and licensed under the MIT License.

---

## 🙋‍♂️ Author

Built by [Your Name] with ❤️ using Python + Streamlit

