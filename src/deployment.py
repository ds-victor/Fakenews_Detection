import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

from src.config import MODEL_SAVE_PATH, TOKENIZER_SAVE_PATH, MAX_SEQ_LEN
from src.preprocessing import clean_text


class FakeNewsService:
    def __init__(self):
        self.model = load_model(MODEL_SAVE_PATH)
        with open(TOKENIZER_SAVE_PATH, "rb") as f:
            self.tokenizer = pickle.load(f)

    def predict(self, text: str):
        cleaned = clean_text(text)

        seq = self.tokenizer.texts_to_sequences([cleaned])
        padded = pad_sequences(seq, maxlen=MAX_SEQ_LEN)

        prob = self.model.predict(padded)[0][0]

        return {
            "label": "REAL" if prob >= 0.5 else "FAKE",
            "probability_real": float(prob)
        }
