import streamlit as st

from src.deployment import FakeNewsService


# ---------- Page Config ----------
st.set_page_config(
    page_title="Fake News Detection",
    page_icon="📰",
    layout="centered"
)

st.title("📰 Fake News Detection App")
st.write(
    """
    This app uses **NLP + BiLSTM + GloVe embeddings**
    to classify news articles as **REAL** or **FAKE**.
    """
)

# ---------- Load Model ----------
@st.cache_resource
def load_model():
    return FakeNewsService()

service = load_model()

# ---------- User Input ----------
text_input = st.text_area(
    "Enter news article text:",
    height=250,
    placeholder="Paste the news content here..."
)

threshold = st.slider(
    "Prediction threshold (REAL vs FAKE)",
    min_value=0.1,
    max_value=0.9,
    value=0.5,
    step=0.05
)

# ---------- Prediction ----------
if st.button("Predict"):
    if not text_input.strip():
        st.warning("Please enter some text before predicting.")
    else:
        with st.spinner("Analyzing news article..."):
            result = service.predict(text_input)

        label = result["label"]
        prob_real = result["probability_real"]

        st.subheader("Prediction Result")

        if label == "REAL":
            st.success(f"REAL News ({prob_real:.2%} confidence)")
        else:
            st.error(f"FAKE News ({1 - prob_real:.2%} confidence)")

        st.write("### Model Confidence")
        st.progress(prob_real if label == "REAL" else 1 - prob_real)
