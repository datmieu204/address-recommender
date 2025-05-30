import numpy as np
from seqeval.metrics import f1_score, classification_report

def align_predictions(predictions, label_ids):
    preds = np.argmax(predictions, axis=2)
    batch_size, seq_len = preds.shape
    out_label_list = [[] for _ in range(batch_size)]
    preds_list = [[] for _ in range(batch_size)]

    for i in range(batch_size):
        for j in range(seq_len):
            if label_ids[i, j] != -100:
                out_label_list[i].append(label_ids[i][j])
                preds_list[i].append(preds[i][j])
    return preds_list, out_label_list

def compute_metrics(p):
    predictions, label_ids = p
    preds_list, out_label_list = align_predictions(predictions, label_ids)
    return {
        "f1": f1_score(out_label_list, preds_list)
    }
