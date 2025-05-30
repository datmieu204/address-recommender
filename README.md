# Address Recommender

The project contains two primary components:

| Package                   | Task                                                                                | Model Backbone                                                                            |
| ------------------------- | ----------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------- |
| `location_classification` | Sentence‐level location classification (does a sentence contain an address?)        | Any 🤗 *transformers* text‑classification model (default: `bert-base-multilingual-cased`) |
| `location_ner`            | Named‑entity recognition for address components (province, district, ward, street…) | Token‑classification model (default: `bert-base-multilingual-cased`)                      |

Both sub‑packages share a **consistent CLI**, configuration files, and helper utilities to streamline training, evaluation, and inference.

---

## ✨ Features

* **Modular codebase** – each concern lives in its own module (`data.py`, `model.py`, `train.py`, …).
* **Zero‑boilerplate training** – run a single CLI command to fine‑tune or evaluate.
* **Config‑driven** – global hyper‑parameters in `config.py` for easy experimentation.
* **Metrics out‑of‑the‑box** – accuracy, F1, confusion‑matrix, seqeval scores.
* **Fast prediction API** – import `LocationClassifier` or `LocationNER` and call `.predict()`.
* **Visualization** – utilities to plot confusion matrices and token‑level predictions.

---

## 📂 Project Structure

```
address-recommender/
├── __init__.py
├── dataset                  # Data crawl and Created Data
├── predict.py               # High‑level inference helpers
├── location_classification/
│   ├── __init__.py
│   ├── cli.py               # Train / evaluate from the command line
│   ├── config.py            # Hyper‑parameters & paths
│   ├── data.py              # Load & split CSV data
│   ├── preprocessing.py     # Tokenisation & Dataset creation
│   ├── model.py             # Model factory
│   ├── train.py             # Trainer wrapper
│   ├── evaluate.py          # Test‑set evaluation
│   ├── metrics.py           # compute_metrics() for Hugging Face Trainer
│   └── visualize.py         # Confusion‑matrix plotting
└── location_ner/
    ├── __init__.py
    ├── cli.py
    ├── config.py
    ├── data.py              # Read CoNLL format
    ├── preprocessing.py
    ├── model.py
    ├── train.py
    └── metrics.py
```

---

## 🚀 Installation

```bash
# 1. Clone / download the repository
unzip address-recommender.zip && cd address-recommender
or
git clone https://github.com/datmieu204/address-recommender

# 2. Create a virtual environment (optional but recommended)
python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -U pip
pip install transformers datasets seqeval scikit-learn matplotlib
```

---

## 📑 Dataset Formats

### 1. Classification

A **CSV** file with at least two columns:

| text                           | label |
| ------------------------------ | ----- |
| "75 Trần Hưng Đạo, Q1, TP.HCM" | 1     |
| "Tôi thích ăn phở."            | 0     |

* **text** – sentence or short paragraph.
* **label** – `1` if the text contains an address, `0` otherwise.

### 2. NER

A **CoNLL‑style** plaintext file (`.txt`, `.conll`, …):

```
75  B-STREET
Trần  I-STREET
Hưng  I-STREET
Đạo  I-STREET
,  O
Q1  B-DISTRICT
,  O
TP.HCM  B-CTY

```

Each line = `token<space>tag`. Sentences separated by a blank line.

---

## 🏃‍♂️ Quick Start

### 1. Train a sentence‑level classifier

```bash
python -m location_classification.cli \
       --csv data/location_sentences.csv \
       --output_dir models/classification
```

### 2. Train an NER model

```bash
python -m location_ner.cli \
       --conll data/location_ner.conll \
       --output_dir models/ner
```

### 3. Evaluate a checkpoint

```bash
python -m location_classification.cli \
       --mode eval \
       --checkpoint_path models/classification/checkpoint-best
```

### 4. Predict in your own script

```python
from address-recommender.predict import LocationClassifier, LocationNER

clf = LocationClassifier(weight_path="models/classification/checkpoint-best")
print(clf("75 Trần Hưng Đạo, Q1, TP.HCM"))  # -> 1

ner = LocationNER(weight_path="models/ner/checkpoint-best")
print(ner("75 Trần Hưng Đạo, Q1, TP.HCM"))
# [('75', 'B-STR'), ('Trần', 'I-STR'), ...]
```

---

## ⚙️ Configuration

Every sub‑package has a `config.py` with sane defaults:

```python
MODEL_NAME = "bert-base-multilingual-cased"
MAX_LENGTH = 128
BATCH_SIZE = 16
EPOCHS = 3
SEED = 42
```

Override via CLI flags or edit the file directly.

---

## 📝 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.