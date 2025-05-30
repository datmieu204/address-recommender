from transformers import AutoTokenizer
from datasets import Dataset

def get_tokenizer(model_name: str):
    return AutoTokenizer.from_pretrained(model_name, use_fast=True)

def encode_dataframe(df, tokenizer, label2id, max_length: int = 128):
    texts = df['address'].tolist()
    encodings = tokenizer(texts, truncation=True, padding=True, max_length=max_length)
    encodings['labels'] = [label2id[l] for l in df['label']]
    return Dataset.from_dict(encodings)
