from sklearn.model_selection import train_test_split

from src.config import *
from src.preprocessing import load_raw_data, add_clean_text_column
from src.model import FakeNewsModel


def main():
    setup_nltk()

    # Load and preprocess data
    df = load_raw_data()
    df = add_clean_text_column(df)

    X = df["clean_text"].values
    y = df["label"].values

    # Train-validation split
    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=SEED,
        stratify=y
    )

    # Build model pipeline
    model = FakeNewsModel(
        max_words=MAX_NUM_WORDS,
        max_len=MAX_SEQ_LEN,
        embed_dim=EMBEDDING_DIM
    )

    model.build_tokenizer(X_train)

    X_train_seq = model.texts_to_sequences(X_train)
    X_val_seq = model.texts_to_sequences(X_val)

    embedding_matrix = model.build_embedding_matrix(GLOVE_PATH)
    model.build_model(embedding_matrix)

    # Train model
    model.model.fit(
        X_train_seq,
        y_train,
        validation_data=(X_val_seq, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE
    )

    # Save artifacts
    MODEL_DIR.mkdir(exist_ok=True)
    model.save(MODEL_SAVE_PATH, TOKENIZER_SAVE_PATH)


if __name__ == "__main__":
    main()
