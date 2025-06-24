# LLM-Length-Estimation

**LLM-Length-Estimation** is a lightweight framework for predicting the decoding length of large language model (LLM) outputs based on the input prompt. This capability is essential for optimizing inference scheduling, batch padding, and reinforcement learning efficiency.

## 🔍 Motivation

In LLM inference and RL training, decoding length varies significantly across prompts. Predicting response length ahead of time helps reduce memory waste and improves scheduling strategies.

---

# 📄 Prepare Dataset
To train a length prediction model, you need a .jsonl dataset with prompt-text and corresponding response length. Each line should follow this format:
```json
{
  "prompt": "Write a short story about a dragon.",
  "length": 128
}
```
You can collect prompt-response pairs from public LLM benchmarks (e.g., LMSYS, AlpacaEval) and compute length as the number of generated tokens.
```shell
python decode_len/data/prepare_lmsys_chat_1m.py
```

Place the file at exps/data/prompt_lengths_k0.jsonl or modify --dataset_path accordingly.

## 🚀 Run a Training Experiment
We provide a BERT-based baseline for prompt-to-length regression:
```shell
python decode_len/train/bert.py --dataset_path exps/data/lmsys_bert_prompt_lengths_k0.jsonl --output_dir exps/results/lmsys_bert_length_predictor_k0 --lr 2e-5 --epochs 10
```