import argparse, random, numpy as np, torch
from datasets import DatasetDict
from . import config, data, preprocessing, model as model_module, train as train_module

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--conll", type=str, default="ner_data.conll")
    parser.add_argument("--model_name", type=str, default=config.MODEL_NAME)
    args = parser.parse_args()

    random.seed(config.SEED)
    np.random.seed(config.SEED)
    torch.manual_seed(config.SEED)

    sentences, tags = data.read_conll(args.conll)
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = data.train_val_test_split(sentences, tags, config.SEED)

    unique_tags = sorted({t for seq in tags for t in seq})
    label2id = {l: i for i, l in enumerate(unique_tags)}
    id2label = {i: l for l, i in label2id.items()}

    tokenizer = preprocessing.get_tokenizer(args.model_name)
    train_ds = preprocessing.encode_examples(X_train, y_train, tokenizer, label2id, config.MAX_LENGTH)
    val_ds = preprocessing.encode_examples(X_val, y_val, tokenizer, label2id, config.MAX_LENGTH)
    test_ds = preprocessing.encode_examples(X_test, y_test, tokenizer, label2id, config.MAX_LENGTH)

    datasets = DatasetDict(train=train_ds, validation=val_ds, test=test_ds)

    model = model_module.get_model(args.model_name, num_labels=len(label2id), id2label=id2label, label2id=label2id)
    trainer = train_module.train(model, datasets['train'], datasets['validation'], tokenizer)
    print(trainer.evaluate(eval_dataset=datasets['test']))

if __name__ == "__main__":
    main()
