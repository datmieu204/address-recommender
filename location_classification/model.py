from transformers import AutoModelForSequenceClassification

def get_model(model_name: str, num_labels: int, id2label=None, label2id=None):
    return AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id,
    )
