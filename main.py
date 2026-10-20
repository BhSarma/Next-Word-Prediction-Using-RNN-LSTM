"""
Next Word Prediction Using RNN, LSTM, GRU and BiLSTM
----------------------------------------------------
python main.py                       -> train (first time) and show sample predictions
python main.py --retrain             -> force retraining
python main.py --predict "Python is" -> quick prediction
python main.py --generate "I love"   -> generate a sentence

Optional: put your own text in corpus.txt (next to this file) to train on it.
"""

import argparse
import json
import os
import re
import time

import numpy as np

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # hide noisy TensorFlow logs

import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import (
    GRU,
    LSTM,
    Bidirectional,
    Dense,
    Dropout,
    Embedding,
    Input,
    SimpleRNN,
)
from tensorflow.keras.metrics import SparseTopKCategoricalAccuracy
from tensorflow.keras.models import Sequential, load_model

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ARTIFACT_DIR = os.path.join(BASE_DIR, "artifacts")
META_PATH = os.path.join(ARTIFACT_DIR, "meta.json")
METRICS_PATH = os.path.join(ARTIFACT_DIR, "metrics.json")
CORPUS_PATH = os.path.join(BASE_DIR, "corpus.txt")

MODEL_NAMES = ["RNN", "LSTM", "GRU", "BiLSTM"]
VERSION = 2          # bump when the saved format changes
MAX_SEQ = 8          # max number of previous words the model looks at
EPOCHS = 150         # early stopping usually stops much sooner
BATCH_SIZE = 16
UNITS = 64
EMB_DIM = 64


def model_path(name):
    return os.path.join(ARTIFACT_DIR, f"{name.lower()}_model.keras")


# ---------------------------------------------------------------- tokenizing
def tokenize(text):
    """Lowercase and split into words."""
    return re.findall(r"[a-z0-9']+", text.lower())


def build_vocab(sentences):
    """Word -> index. 0 = padding, 1 = unknown word."""
    word_index = {"<unk>": 1}
    for sentence in sentences:
        for word in tokenize(sentence):
            if word not in word_index:
                word_index[word] = len(word_index) + 1
    return word_index


def texts_to_ids(text, word_index):
    """Convert text to ids; unknown words are skipped."""
    return [word_index[w] for w in tokenize(text) if w in word_index and w != "<unk>"]


def pad_pre(sequences, maxlen):
    """Left-pad (and left-truncate) sequences with zeros."""
    out = np.zeros((len(sequences), maxlen), dtype="int32")
    for i, seq in enumerate(sequences):
        seq = seq[-maxlen:]
        if seq:
            out[i, -len(seq):] = seq
    return out


# ------------------------------------------------------------------- dataset
def load_corpus():
    """Use corpus.txt if it exists, otherwise the sentences in data.py."""
    if os.path.exists(CORPUS_PATH):
        with open(CORPUS_PATH, encoding="utf-8") as f:
            raw = f.read()
        parts = re.split(r"[.!?\n]+", raw)
        sentences = [p.strip() for p in parts if len(tokenize(p)) >= 2]
        if sentences:
            print(f"Using {len(sentences)} sentences from corpus.txt")
            return sentences
    from data import SENTENCES

    print(f"Using {len(SENTENCES)} built-in sentences from data.py")
    return SENTENCES


def prepare_data(sentences):
    word_index = build_vocab(sentences)
    total_words = len(word_index) + 1

    sequences = []
    for sentence in sentences:
        ids = texts_to_ids(sentence, word_index)
        for i in range(1, len(ids)):
            sequences.append(ids[max(0, i - MAX_SEQ): i + 1])

    max_length = max(len(s) for s in sequences)
    padded = pad_pre(sequences, max_length)
    X = padded[:, :-1]
    y = padded[:, -1]  # integer labels (sparse), no huge one-hot matrix

    # train / validation split (90 / 10)
    rng = np.random.default_rng(42)
    idx = rng.permutation(len(X))
    n_val = max(1, int(0.1 * len(X)))
    val_idx, train_idx = idx[:n_val], idx[n_val:]
    return (
        X[train_idx], y[train_idx], X[val_idx], y[val_idx],
        word_index, total_words, max_length,
    )


# ------------------------------------------------------------------- models
def build_model(kind, total_words, seq_len):
    if kind == "RNN":
        core = SimpleRNN(UNITS)
    elif kind == "LSTM":
        core = LSTM(UNITS)
    elif kind == "GRU":
        core = GRU(UNITS)
    else:
        core = Bidirectional(LSTM(UNITS))
    model = Sequential(
        [
            Input(shape=(seq_len,)),
            Embedding(input_dim=total_words, output_dim=EMB_DIM),
            core,
            Dropout(0.2),
            Dense(total_words, activation="softmax"),
        ]
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.003),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy", SparseTopKCategoricalAccuracy(k=3, name="top3")],
    )
    return model


def train_and_save():
    tf.keras.utils.set_random_seed(42)
    sentences = load_corpus()
    Xtr, ytr, Xval, yval, word_index, total_words, max_length = prepare_data(sentences)
    seq_len = max_length - 1
    os.makedirs(ARTIFACT_DIR, exist_ok=True)
    print(f"Vocabulary: {total_words - 2} words | train samples: {len(Xtr)} | val samples: {len(Xval)}")

    all_metrics = {}
    for name in MODEL_NAMES:
        print(f"\n=== Training {name} ===")
        start = time.time()
        model = build_model(name, total_words, seq_len)
        hist = model.fit(
            Xtr, ytr,
            validation_data=(Xval, yval),
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            callbacks=[EarlyStopping(monitor="val_loss", patience=30, restore_best_weights=True)],
            verbose=0,
        )
        train_res = model.evaluate(Xtr, ytr, verbose=0, return_dict=True)
        val_res = model.evaluate(Xval, yval, verbose=0, return_dict=True)
        model.save(model_path(name))

        all_metrics[name] = {
            "train_acc": float(train_res["accuracy"]),
            "val_acc": float(val_res["accuracy"]),
            "val_top3": float(val_res["top3"]),
            "val_loss": float(val_res["loss"]),
            "perplexity": float(np.exp(val_res["loss"])),
            "epochs_run": len(hist.history["loss"]),
            "seconds": round(time.time() - start, 1),
            "history": {k: [float(v) for v in vals] for k, vals in hist.history.items()
                        if k in ("loss", "val_loss", "accuracy", "val_accuracy")},
        }
        m = all_metrics[name]
        print(f"{name}: train acc {m['train_acc']:.2f} | val acc {m['val_acc']:.2f} | "
              f"top-3 {m['val_top3']:.2f} | perplexity {m['perplexity']:.1f} | "
              f"{m['epochs_run']} epochs, {m['seconds']}s")

    with open(METRICS_PATH, "w") as f:
        json.dump(all_metrics, f)
    with open(META_PATH, "w") as f:
        json.dump({"version": VERSION, "word_index": word_index, "max_length": max_length}, f)
    print(f"\nSaved models, vocabulary and metrics to: {ARTIFACT_DIR}")


def artifacts_ready():
    if not os.path.exists(META_PATH):
        return False
    try:
        with open(META_PATH) as f:
            if json.load(f).get("version") != VERSION:
                return False
    except Exception:
        return False
    return all(os.path.exists(model_path(n)) for n in MODEL_NAMES)


# --------------------------------------------------------------- prediction
class Predictor:
    """Loads saved artifacts, predicts the next word and generates text."""

    def __init__(self):
        if not artifacts_ready():
            raise FileNotFoundError("No trained models found. Run `python main.py --retrain` first.")
        with open(META_PATH) as f:
            meta = json.load(f)
        self.word_index = meta["word_index"]
        self.index_word = {i: w for w, i in self.word_index.items()}
        self.seq_len = meta["max_length"] - 1
        self.models = {n: load_model(model_path(n)) for n in MODEL_NAMES}
        self.rng = np.random.default_rng()

    def _probs(self, ids, model_name):
        x = pad_pre([ids], self.seq_len)
        p = self.models[model_name](x, training=False).numpy()[0].astype("float64")
        p[0] = 0.0  # padding
        p[1] = 0.0  # unknown
        return p / p.sum()

    def predict(self, text, model_name="LSTM", top_k=3):
        """Return a list of (word, probability), or [] if no known words."""
        ids = texts_to_ids(text, self.word_index)
        if not ids:
            return []
        p = self._probs(ids, model_name)
        top = np.argsort(p)[::-1][:top_k]
        return [(self.index_word[int(i)], float(p[i])) for i in top]

    def generate(self, text, model_name="LSTM", n_words=8, temperature=0.7):
        """Keep predicting the next word. temperature 0 = always the best word."""
        ids = texts_to_ids(text, self.word_index)
        if not ids:
            return ""
        new_words = []
        for _ in range(n_words):
            p = self._probs(ids, model_name)
            if temperature <= 0.05:
                nxt = int(np.argmax(p))
            else:
                q = np.power(p + 1e-12, 1.0 / temperature)
                q /= q.sum()
                nxt = int(self.rng.choice(len(q), p=q))
            ids.append(nxt)
            new_words.append(self.index_word[nxt])
        return " ".join(tokenize(text) + new_words)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predict", type=str, help="Text to predict next word for")
    parser.add_argument("--generate", type=str, help="Text to start generating from")
    parser.add_argument("--retrain", action="store_true", help="Force retraining")
    args = parser.parse_args()

    if args.retrain or not artifacts_ready():
        train_and_save()

    predictor = Predictor()

    if args.generate:
        for name in MODEL_NAMES:
            print(f"[{name}] {predictor.generate(args.generate, name)}")
        return

    tests = [args.predict] if args.predict else ["I", "Python is", "Deep learning"]
    for text in tests:
        for name in MODEL_NAMES:
            result = predictor.predict(text, name, top_k=1)
            if result:
                print(f"[{name}] '{text}' -> {result[0][0]} ({result[0][1]:.2f})")
            else:
                print(f"[{name}] '{text}' -> no known words in input")


if __name__ == "__main__":
    main()