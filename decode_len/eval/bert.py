import json
import torch
from typing import Callable, List, Dict
from tqdm import tqdm
from transformers import set_seed
from transformers import BertTokenizer, BertForSequenceClassification

from decode_len.eval.metrics import *
from decode_len.predictor.bert import load_model, predict

def evaluate(dataset_path: str, predictor: Callable[[str], int], tolerance: int = 5) -> Dict[str, float]:
    def _load_jsonl_subset(path: str, num_samples: str) -> List[Dict]:
        with open(path) as f:
            lines = [json.loads(line) for line in f]
        if num_samples.endswith("%"):
            pct = float(num_samples.strip('%')) / 100
            k = int(len(lines) * pct)
        else:
            k = int(num_samples)
        return lines[:k]
    
    golds = []
    preds = []
    samples = _load_jsonl_subset(dataset_path, args.num_samples)
    for item in tqdm(samples, desc="Evaluating"):
        prompt = item["prompt"]
        gold_len = item["length"]
        pred_len = predictor(prompt)

        golds.append(gold_len)
        preds.append(pred_len)

    return {
        "accuracy": accuracy(golds, preds),
        f"accuracy_±{tolerance}": accuracy_with_tolerance(golds, preds, tolerance),
        "smape": smape(golds, preds),
        "kendall_tau": kendall_tau(preds, golds),
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, default="exps/data/prompt_lengths_k0.jsonl", help="Path to jsonl dataset with prompt and length")
    parser.add_argument("--num_samples", type=str, default="100%")  # support "1%", or "1000"
    parser.add_argument("--tolerance", type=int, default=5, help="±k token tolerance for relaxed accuracy")
    args = parser.parse_args()

    set_seed(42)
    load_model()
    metrics = evaluate(args.dataset_path, predict, tolerance=args.tolerance)

    print("\nEvaluation Results:")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")
