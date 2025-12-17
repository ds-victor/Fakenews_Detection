# Imports
import numpy as np
import pickle

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.optimizers import Adam


class FakeNewsModel:
    def __init__(self, max_words, max_len, embed_dim):
        self.max_words = max_words
        self.max_len = max_len
        self.embed_dim = embed_dim
        self.tokenizer = None
        self.model = None

    def build_tokenizer(self, texts):
        self.tokenizer = Tokenizer(
            num_words=self.max_words,
            oov_token="<OOV>"
        )
        self.tokenizer.fit_on_texts(texts)

    def texts_to_sequences(self, texts):
        sequences = self.tokenizer.texts_to_sequences(texts)
        return pad_sequences(
            sequences,
            maxlen=self.max_len,
            padding="post",
            truncating="post"
        )

    def load_glove_embeddings(self, glove_path):
        embeddings = {}
        with open(glove_path, encoding="utf-8") as f:
            for line in f:
                values = line.split()
                word = values[0]
                vector = np.asarray(values[1:], dtype="float32")
                embeddings[word] = vector
        return embeddings

    def build_embedding_matrix(self, glove_path):
        embeddings = self.load_glove_embeddings(glove_path)
        matrix = np.random.normal(
            size=(self.max_words, self.embed_dim)
        )

        for word, idx in self.tokenizer.word_index.items():
            if idx < self.max_words and word in embeddings:
                matrix[idx] = embeddings[word]

        return matrix

    def build_model(self, embedding_matrix):
        self.model = Sequential([
            Embedding(
                input_dim=self.max_words,
                output_dim=self.embed_dim,
                weights=[embedding_matrix],
                input_length=self.max_len,
                trainable=False
            ),
            Bidirectional(LSTM(128)),
            Dropout(0.5),
            Dense(64, activation="relu"),
            Dropout(0.5),
            Dense(1, activation="sigmoid")
        ])

        self.model.compile(
            loss="binary_crossentropy",
            optimizer=Adam(learning_rate=0.001),
            metrics=["accuracy"]
        )

    def save(self, model_path, tokenizer_path):
        self.model.save(model_path)
        with open(tokenizer_path, "wb") as f:
            pickle.dump(self.tokenizer, f)
