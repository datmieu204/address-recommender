import argparse
import numpy as np, random, torch, os
from datasets import DatasetDict
from . import config, data, preprocessing, model as model_module, train as train_module, evaluate as evaluate_module

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, default="combined_location_data.csv")
    parser.add_argument("--model_name", type=str, default=config.MODEL_NAME)
    args = parser.parse_args()

    # Reproducibility
    random.seed(config.SEED)
    np.random.seed(config.SEED)
    torch.manual_seed(config.SEED)

    df = data.load_data(args.csv)
    train_df, val_df, test_df = data.train_val_test_split(df, config.SEED)
    label2id, id2label = data.get_label_mapping(train_df)

    tokenizer = preprocessing.get_tokenizer(args.model_name)
    train_ds = preprocessing.encode_dataframe(train_df, tokenizer, label2id, config.MAX_LENGTH)
    val_ds = preprocessing.encode_dataframe(val_df, tokenizer, label2id, config.MAX_LENGTH)
    test_ds = preprocessing.encode_dataframe(test_df, tokenizer, label2id, config.MAX_LENGTH)

    tokenized = DatasetDict(train=train_ds, validation=val_ds, test=test_ds)

    model = model_module.get_model(args.model_name, num_labels=len(label2id), id2label=id2label, label2id=label2id)
    trainer = train_module.train(model, tokenized['train'], tokenized['validation'], tokenizer)
    metrics = evaluate_module.evaluate(trainer, tokenized['test'])
    print(metrics)

if __name__ == "__main__":
    main()
