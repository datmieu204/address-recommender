"""High-level prediction utilities for location classification & NER."""
import torch
from transformers import AutoTokenizer
from location_classification import model as cls_model_module, preprocessing as cls_preproc, config as cls_config
from location_ner import model as ner_model_module, preprocessing as ner_preproc, config as ner_config

class LocationClassifier:
    def __init__(self, model_ckpt: str = cls_config.MODEL_NAME, weight_path: str | None = None):
        self.tokenizer = cls_preproc.get_tokenizer(model_ckpt)
        self.model = cls_model_module.get_model(model_ckpt, num_labels=1) if weight_path is None else cls_model_module.get_model(model_ckpt, num_labels=1).from_pretrained(weight_path)
        self.model.eval()

    @torch.no_grad()
    def __call__(self, texts):
        if isinstance(texts, str):
            texts = [texts]
        enc = self.tokenizer(texts, truncation=True, padding=True, max_length=cls_config.MAX_LENGTH, return_tensors="pt")
        logits = self.model(**enc).logits
        preds = logits.argmax(dim=-1).tolist()
        return preds

class LocationNER:
    def __init__(self, model_ckpt: str = ner_config.MODEL_NAME, weight_path: str | None = None, id2label: dict[str,int] | None = None):
        self.tokenizer = ner_preproc.get_tokenizer(model_ckpt)
        self.model = ner_model_module.get_model(model_ckpt, num_labels=len(id2label) if id2label else 1) if weight_path is None else ner_model_module.get_model(model_ckpt, num_labels=len(id2label)).from_pretrained(weight_path)
        self.id2label = id2label
        self.model.eval()

    @torch.no_grad()
    def __call__(self, sentence_tokens):
        # sentence_tokens: list[str]
        enc = self.tokenizer(sentence_tokens, is_split_into_words=True, return_tensors="pt")
        outputs = self.model(**enc)
        preds = outputs.logits.argmax(dim=-1).squeeze().tolist()
        words = sentence_tokens
        tags = [self.id2label[p] if p in self.id2label else 'O' for p in preds]
        return list(zip(words, tags))
