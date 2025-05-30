import pandas as pd
from sklearn.model_selection import train_test_split

def read_conll(path):
    """Read data in CoNLL format with columns: token, tag"""
    sentences, tags, cur_tokens, cur_tags = [], [], [], []
    with open(path, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                if cur_tokens:
                    sentences.append(cur_tokens)
                    tags.append(cur_tags)
                    cur_tokens, cur_tags = [], []
            else:
                tok, tag = line.split()
                cur_tokens.append(tok)
                cur_tags.append(tag)
    if cur_tokens:
        sentences.append(cur_tokens)
        tags.append(cur_tags)
    return sentences, tags

def train_val_test_split(sentences, tags, seed=25):
    from sklearn.model_selection import train_test_split
    X_train, X_temp, y_train, y_temp = train_test_split(sentences, tags, test_size=0.2, random_state=seed)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=seed)
    return (X_train, y_train), (X_val, y_val), (X_test, y_test)
