import argparse
import torch
from transformers import BertTokenizer, BertForSequenceClassification

def load_model(model_dir, device):
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    model = BertForSequenceClassification.from_pretrained(model_dir)
    model.to(device)
    model.eval()
    return tokenizer, model

def predict(prompt, tokenizer, model, max_len, device):
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, padding="max_length", max_length=max_len)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
        predicted_class = torch.argmax(outputs.logits, dim=-1).item()
    return predicted_class

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    tokenizer, model = load_model(args.model_dir, device)

    if args.prompt:
        # single prompt predict
        pred_len = predict(args.prompt, tokenizer, model, args.max_len, device)
        print(f"Predicted response length: {pred_len}")
    elif args.input_file:
        # batch prompt predict
        with open(args.input_file) as f:
            prompts = [line.strip() for line in f if line.strip()]
        for i, prompt in enumerate(prompts):
            pred_len = predict(prompt, tokenizer, model, args.max_len, device)
            print(f"[{i}] {prompt[:50]}... => length: {pred_len}")
    else:
        print("Please provide either --prompt or --input_file")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, default="len_pred/bert_length_predictor")
    parser.add_argument("--max_len", type=int, default=128)
    parser.add_argument("--prompt", type=str, default=None, help="A single prompt string to predict length for.")
    parser.add_argument("--input_file", type=str, default=None, help="A file containing prompts (one per line).")
    parser.add_argument("--cpu", action="store_true", help="Force CPU even if GPU is available.")
    args = parser.parse_args()
    main(args)
