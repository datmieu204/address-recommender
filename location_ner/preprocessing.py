from transformers import AutoTokenizer
from datasets import Dataset
import itertools

def get_tokenizer(model_name):
    return AutoTokenizer.from_pretrained(model_name, use_fast=True)

def encode_examples(sentences, tags, tokenizer, label2id, max_length=128):
    encodings = tokenizer(sentences,
                          is_split_into_words=True,
                          truncation=True,
                          padding=True,
                          max_length=max_length)
    labels = []
    for i, label in enumerate(tags):
        word_ids = encodings.word_ids(batch_index=i)
        label_ids = []
        previous_word_idx = None
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:
                label_ids.append(label2id[label[word_idx]])
            else:
                label_ids.append(label2id[label[word_idx]] if label[word_idx].startswith('B') else label2id[label[word_idx].replace('B-', 'I-')])
            previous_word_idx = word_idx
        labels.append(label_ids)
    encodings['labels'] = labels
    return Dataset.from_dict(encodings)
