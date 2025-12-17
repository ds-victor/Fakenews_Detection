# Imports
import re
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from src.config import FAKE_PATH, TRUE_PATH

lemmatizer = WordNetLemmatizer()


def get_stopwords():
    return set(stopwords.words("english"))


def load_raw_data() -> pd.DataFrame:
    fake = pd.read_csv(FAKE_PATH)
    true = pd.read_csv(TRUE_PATH)

    fake["label"] = 0
    true["label"] = 1

    df = pd.concat([fake, true], axis=0)
    df = df.sample(frac=1).reset_index(drop=True)

    return df


def clean_text(text: str) -> str:
    stop_words = get_stopwords()

    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"[^a-zA-Z]", " ", text)

    tokens = nltk.word_tokenize(text)

    tokens = [
        lemmatizer.lemmatize(w)
        for w in tokens
        if w not in stop_words and len(w) > 2
    ]

    return " ".join(tokens)


def add_clean_text_column(df: pd.DataFrame) -> pd.DataFrame:
    df["clean_text"] = df["text"].apply(clean_text)
    return df
