import os
import json, argparse
from tqdm import tqdm
from datasets import load_dataset, concatenate_datasets
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

def main(args):
    train_set = load_dataset(args.dataset_name, name="main", split="train")
    test_set = load_dataset(args.dataset_name, name="main", split="test")
    dataset = concatenate_datasets([train_set, test_set])
    
    # If needs, load LLM
    if args.use_llm:
        tokenizer = AutoTokenizer.from_pretrained(args.model_id)
        model = AutoModelForCausalLM.from_pretrained(args.model_id)
        generator = pipeline("text-generation", model=model, tokenizer=tokenizer, device=args.device)
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_id or args.model_id)

    output = []
    for example in tqdm(dataset):
        try:
            prompt = example["question"]

            if args.use_llm:
                response = generator(prompt, max_new_tokens=args.max_new_tokens, do_sample=False)[0]["generated_text"]
                response_only = response[len(prompt):].strip()
            else:
                response_only = example["answer"]

            # Tokenize and slice response
            response_tokens = tokenizer.encode(response_only, add_special_tokens=False)
            if len(response_tokens) <= args.k:
                continue  # Skip if response too short

            prefix_response = tokenizer.decode(response_tokens[:args.k], skip_special_tokens=True)
            modified_prompt = prompt + prefix_response
            target_length = len(response_tokens) - args.k
            output.append({"prompt": modified_prompt, "length": target_length})

        except Exception as e:
            print(f"Error on prompt: {prompt[:50]}... {e}")
            continue

    with open(args.output_path, "w") as f:
        for item in output:
            f.write(json.dumps(item) + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, default="openai/gsm8k")
    parser.add_argument("--num_samples", type=str, default="100%")  # support "1%", or "1000"
    parser.add_argument("--use_llm", action="store_true", help="Use LLM to generate response or not")
    parser.add_argument("--model_id", type=str, default="Qwen/Qwen3-8B", help="LLM for generating response, only valid when use_llm=True")
    parser.add_argument("--tokenizer_id", type=str, default=None, help="tokenizer (default the same to model_id)")
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--output_path", type=str, default="exps/data/gsm8k_bert_prompt_lengths.jsonl")
    parser.add_argument("--k", type=int, default=0, help="Number of response tokens to append to prompt")
    args = parser.parse_args()
    if args.output_path is not None:
        if f"_k{args.k}" not in args.output_path:
                args.output_path = args.output_path.rsplit(".", 1)[0] + f"_k{args.k}.jsonl"
    else:
        args.output_path = f"exps/data/prompt_lengths_k{args.k}.jsonl"
    dir_path = os.path.dirname(args.output_path)
    os.makedirs(dir_path, exist_ok=True)
    main(args)
