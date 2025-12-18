# 📰 Fake News Detection using NLP, LSTM & GloVe

A production-ready Fake News Detection system built using Natural Language Processing (NLP) and a Bi-Directional LSTM model with GloVe word embeddings, deployed as an interactive Streamlit web application.

## 🔗 Live App:
👉 https://fakenewsdetection-vtqt7hkwdt5soeiiesp5qn.streamlit.app/

## 📌 Problem Statement

With the rapid spread of misinformation online, identifying fake news articles has become a critical challenge.
This project aims to automatically classify news articles as REAL or FAKE using deep learning–based NLP techniques.

## 🚀 Key Features
- NLP preprocessing using NLTK
- Text representation using pre-trained GloVe embeddings
- Bi-Directional LSTM for sequence learning
- Clean separation of:
    - Data preprocessing
    - Model training
    - Inference logic
    - UI layer
- Streamlit Cloud deployment for real-time predictions

## 🌐 Live Application
The trained model is deployed using Streamlit Cloud.
### 🔗 Try the app here:
👉 https://fakenewsdetection-vtqt7hkwdt5soeiiesp5qn.streamlit.app/

## App Capabilities
- Paste a news article
- Get instant REAL / FAKE prediction
- View model confidence score

## 🧠 Model Architecture
```
Input Text
   ↓
Text Cleaning (NLTK)
   ↓
Tokenization & Padding
   ↓
GloVe Embedding Layer (100-dim)
   ↓
Bi-Directional LSTM
   ↓
Dense Layers
   ↓
Binary Classification (REAL / FAKE)

```
## 📁 Project Structure
```
fakenews_detection/
│
├── app.py              👈 Streamlit entry point (ROOT)
│
├── src/
│   ├── __init__.py
│   ├── deployment.py
│   ├── config.py
│   ├── preprocessing.py
│   ├── model.py
│   └── training.py
│
├── models/
├── requirements.txt
├── README.md

```
## 📊 Dataset
- Fake.csv – Fake news articles
- True.csv – Real news articles
These are commonly used public datasets for fake news classification tasks.

## 🔹 GloVe Embeddings (Training Only)

This project uses GloVe Twitter embeddings (100-dim) during training.

📥 Download from:
https://nlp.stanford.edu/projects/glove/

Required file:
```
glove.twitter.27B.100d.txt

```
📂 Place it inside:
```
data/
```
⚠️ Note:
The GloVe file (~1GB) is NOT included in this repository due to GitHub size limits.
It is only required for training, not for deployment.

## ⚙️ Installation & Setup (Local)
### 1️⃣ Create virtual environment
```
python -m venv venv
venv\Scripts\activate   # Windows
```
### 2️⃣ Install dependencies
```
pip install -r requirements.txt
```
### 3️⃣ Download NLTK resources (one-time)
```
import nltk
nltk.download("punkt")
nltk.download("stopwords")
nltk.download("wordnet")
```
## 🏋️ Model Training

Run training from the project root:
```
python -m src.training
```
This will:
- Preprocess text
- Train the Bi-LSTM model
- Save model & tokenizer in models/

## 🌐 Run Streamlit App Locally
From project root:
```
streamlit run app.py
```
Open browser at:
```
http://localhost:8501
```
## 🧪 Example Output

{
  "label": "FAKE",
  "probability_real": 0.14
}

## 🧠 Design Decisions & Best Practices
- Single import style (from src...) across the project
- No training in Streamlit app (predict-only deployment)
- Large files (GloVe, NLTK data) excluded via .gitignore
- Model artifacts committed for Streamlit Cloud inference
- Clear separation between experimentation (notebooks) and production code

## 📌 Future Enhancements
- Attention mechanism
- Transformer-based models (BERT)
- Model explainability (LIME / SHAP)
- FastAPI backend
- Docker deployment


