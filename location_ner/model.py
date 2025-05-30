from transformers import AutoModelForTokenClassification

def get_model(model_name, num_labels, id2label=None, label2id=None):
    return AutoModelForTokenClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id,
    )
