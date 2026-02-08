import torch
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
import json
import numpy as np
from sklearn.metrics import roc_auc_score
import os


def create_dummy_dataset(file_path):
    if not os.path.exists(file_path):
        print(f"Dataset file '{file_path}' not found. Creating a dummy dataset.")
        data = {
            "Human": [
            ],
            "AI": [
            ]
        }
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)


def calculate_perplexity(text, model, tokenizer):
    if not text.strip():
        return float('inf')
    encodings = tokenizer(text, return_tensors='pt', truncation=True, max_length=1024)
    input_ids = encodings.input_ids.to(model.device)
    if input_ids.size(1) < 2:
        return float('inf')
    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
        neg_log_likelihood = outputs.loss
    ppl = torch.exp(neg_log_likelihood)
    return ppl.item()


def evaluate_and_get_auroc(file_path, model, tokenizer):
    create_dummy_dataset(file_path)
    print(f"Loading dataset from {file_path}...")
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    human_texts = data.get("Human", [])
    ai_texts = data.get("AI", [])

    true_labels = [0] * len(human_texts) + [1] * len(ai_texts)
    scores = []

    print("Calculating perplexity for human texts...")
    for i, text in enumerate(human_texts):
        ppl = calculate_perplexity(text, model, tokenizer)
        scores.append(ppl)
        print(f"  - [Human {i + 1}] PPL: {ppl:<8.2f} Text: '{text[:40]}...'")

    print("\nCalculating perplexity for AI texts...")
    for i, text in enumerate(ai_texts):
        ppl = calculate_perplexity(text, model, tokenizer)
        scores.append(ppl)
        print(f"  - [AI {i + 1}]    PPL: {ppl:<8.2f} Text: '{text[:40]}...'")

    prediction_scores = -np.array(scores)
    auroc_score = roc_auc_score(true_labels, prediction_scores)
    return auroc_score, scores[:len(human_texts)], scores[len(human_texts):]


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    print("Loading pre-trained model (gpt2)... This may take a moment.")
    model_id = "gpt2"
    model = GPT2LMHeadModel.from_pretrained(model_id).to(device)
    model.eval()
    tokenizer = GPT2TokenizerFast.from_pretrained(model_id)

    dataset_file = "text_dataset.json"
    auroc, human_scores, ai_scores = evaluate_and_get_auroc(dataset_file, model, tokenizer)

    print("\n" + "=" * 50)
    print(" " * 15 + "Classification Results")
    print("=" * 50)
    print(f"\nAverage Perplexity for HUMAN texts: {np.mean(human_scores):.2f}")
    print(f"Average Perplexity for AI texts:    {np.mean(ai_scores):.2f}\n")
    print(f"AUROC Score: {auroc:.4f}")
    print("=" * 50)
    if auroc > 0.8:
        print("\nConclusion: The model shows a strong ability to distinguish.")
    else:
        print("\nConclusion: The model shows some ability to distinguish.")


if __name__ == "__main__":
    main()