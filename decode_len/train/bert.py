import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import numpy as np
import json, argparse, torch
from torch.utils.data import Dataset, Subset
from transformers import (
    set_seed,
    BertTokenizer, BertForSequenceClassification,
    Trainer, TrainingArguments,
    DataCollatorWithPadding
)
from sklearn.model_selection import train_test_split

from decode_len.data.dataset import PromptLengthDataset
from decode_len.eval.metrics import *

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = logits.argmax(-1)
    return {
        "accuracy": accuracy(preds, labels),
        "accuracy_with_tolerance": accuracy_with_tolerance(preds, labels),
        "kendall_tau": kendall_tau(preds, labels),
        "smape": smape(preds, labels),
    }

def freeze_bert_except_classifier(model):
    for _, param in model.bert.named_parameters():
        param.requires_grad = False
    for _, param in model.classifier.named_parameters():
        param.requires_grad = True
    return model

def main(args):
    set_seed(42)
    tokenizer = BertTokenizer.from_pretrained(args.bert_model)
    model = BertForSequenceClassification.from_pretrained(args.bert_model, num_labels=args.max_length)
    # model = freeze_bert_except_classifier(model)

    full_dataset = PromptLengthDataset(args.dataset_path, tokenizer, args.max_len, args.max_length)

    # Split train/val
    indices = list(range(len(full_dataset)))
    train_idx, val_idx = train_test_split(indices, test_size=0.01, random_state=42)
    train_dataset = Subset(full_dataset, train_idx)
    val_dataset = Subset(full_dataset, val_idx)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        per_device_eval_batch_size=args.batch_size,
        eval_strategy=args.evaluation_strategy,
        eval_steps=args.eval_steps,
        save_strategy=args.save_strategy,
        save_total_limit=args.save_total_limit,
        save_steps=args.save_steps,
        load_best_model_at_end=True,
        metric_for_best_model=args.metric_for_best_model,
        greater_is_better=args.greater_is_better,
        logging_dir=args.logging_dir or f"{args.output_dir}/logs",
        logging_steps=args.logging_steps,
        lr_scheduler_type = "linear",
        warmup_ratio = 0.1,
        torch_compile=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--bert_model", type=str, default="bert-base-uncased")
    parser.add_argument("--dataset_path", type=str, default="len_pred/prompt_lengths.jsonl")
    parser.add_argument("--output_dir", type=str, default="len_pred/bert_length_predictor")
    parser.add_argument("--evaluation_strategy", type=str, default="steps")
    parser.add_argument("--eval_steps", type=int, default=1000)
    parser.add_argument("--save_strategy", type=str, default="steps")
    parser.add_argument("--save_total_limit", type=int, default=2)
    parser.add_argument("--save_steps", type=int, default=1000)
    parser.add_argument("--metric_for_best_model", type=str, default="eval_loss")
    parser.add_argument("--greater_is_better", action="store_true")
    parser.add_argument("--logging_dir", type=str, default="./logs")
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--max_len", type=int, default=128)
    parser.add_argument("--max_length", type=int, default=1024)  # max predicted token length
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=10)
    args = parser.parse_args()
    main(args)