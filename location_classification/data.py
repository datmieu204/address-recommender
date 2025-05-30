import pandas as pd
from sklearn.model_selection import train_test_split

def load_data(path: str) -> pd.DataFrame:
    """Load CSV dataset containing 'address' and 'label' columns."""
    return pd.read_csv(path, encoding='utf-8')

def train_val_test_split(df, seed: int = 25):
    """20% temp, then split into 10% val + 10% test."""
    train_df, temp = train_test_split(df, test_size=0.2, random_state=seed, stratify=df['label'])
    val_df, test_df = train_test_split(temp, test_size=0.5, random_state=seed, stratify=temp['label'])
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True), test_df.reset_index(drop=True)

def get_label_mapping(df):
    labels = sorted(df['label'].unique())
    label2id = {l: i for i, l in enumerate(labels)}
    id2label = {i: l for l, i in label2id.items()}
    return label2id, id2label
