from pathlib import Path
import nltk

# Project root
BASE_DIR = Path(__file__).resolve().parent.parent

# Data paths
DATA_DIR = BASE_DIR / "data"
FAKE_PATH = DATA_DIR / "Fake.csv"
TRUE_PATH = DATA_DIR / "True.csv"
GLOVE_PATH = DATA_DIR / "glove.twitter.27B.100d.txt"

# Model paths
MODEL_DIR = BASE_DIR / "models"
MODEL_SAVE_PATH = MODEL_DIR / "fake_news_lstm.h5"
TOKENIZER_SAVE_PATH = MODEL_DIR / "tokenizer.pkl"

# NLP parameters
MAX_NUM_WORDS = 20000
MAX_SEQ_LEN = 300
EMBEDDING_DIM = 100

# Training parameters
TEST_SIZE = 0.2
BATCH_SIZE = 128
EPOCHS = 5
SEED = 42

# NLTK setup
NLTK_DATA_DIR = BASE_DIR / "nltk_data"
nltk.data.path.append(str(NLTK_DATA_DIR))


def setup_nltk():
    resources = ["punkt", "stopwords", "wordnet"]
    for r in resources:
        try:
            nltk.data.find(r)
        except LookupError:
            nltk.download(r, download_dir=NLTK_DATA_DIR)


