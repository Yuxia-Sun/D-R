import torch
from torch.distributions import Categorical
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
import json
import numpy as np
from sklearn.metrics import roc_auc_score
import os
torch.backends.cuda.matmul.allow_tf32 = True
torch.set_grad_enabled(False)

def create_dummy_dataset(file_path):
    if not os.path.exists(file_path):
        print(f"Dataset file '{file_path}' not found. Creating a dummy dataset.")
        data = {"Human": [], "AI": []}
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

def batch_entropy(texts, model, tokenizer, device, max_length=1024,
                  init_batch_size=64, amp=True):
    if not texts:
        return []

    entropies = []
    i = 0
    bsz = init_batch_size

    while i < len(texts):
        tried = False
        while True:
            try:
                batch = texts[i:i+bsz]
                inputs = tokenizer(
                    batch,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=max_length
                )
                input_ids = inputs["input_ids"].to(device, non_blocking=True)
                attention_mask = inputs["attention_mask"].to(device, non_blocking=True)

                if amp and device.startswith("cuda"):
                    autocast_ctx = torch.autocast(device_type="cuda", dtype=torch.float16)
                else:
                    class DummyCtx:
                        def __enter__(self): return None
                        def __exit__(self, exc_type, exc, tb): return False
                    autocast_ctx = DummyCtx()

                with torch.inference_mode(), autocast_ctx:
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                    logits = outputs.logits  # [B, L, V]

                logits = logits[:, :-1, :]
                attn = attention_mask[:, 1:]

                dist = Categorical(logits=logits)
                token_entropy = dist.entropy()       # [B, L-1]

                denom = attn.sum(dim=1).clamp_min(1)
                seq_entropy = (token_entropy * attn).sum(dim=1) / denom
                entropies.extend(seq_entropy.detach().float().cpu().tolist())

                i += bsz
                if tried and bsz < init_batch_size:
                    bsz = min(init_batch_size, bsz * 2)
                break

            except RuntimeError as e:
                if "out of memory" in str(e).lower() and bsz > 1:
                    torch.cuda.empty_cache()
                    bsz = max(1, bsz // 2)
                    tried = True
                    print(f"[Warn] CUDA OOM: reduce batch_size to {bsz}")
                    continue
                else:
                    raise e

    return entropies


def evaluate_and_get_auroc_entropy(file_path, model, tokenizer, device,
                                   max_length=1024, init_batch_size=64, amp=True):
    create_dummy_dataset(file_path)

    print(f"Loading dataset from {file_path}...")
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    human_texts = data.get("Human", [])
    ai_texts = data.get("AI", [])

    if len(human_texts) == 0 or len(ai_texts) == 0:
        return None, [], []

    print(f"Num Human: {len(human_texts)} | Num AI: {len(ai_texts)}")

    print("Calculating average entropy for human texts (batched)...")
    human_scores = batch_entropy(
        human_texts, model, tokenizer, device,
        max_length=max_length, init_batch_size=init_batch_size, amp=amp
    )

    print("Calculating average entropy for AI texts (batched)...")
    ai_scores = batch_entropy(
        ai_texts, model, tokenizer, device,
        max_length=max_length, init_batch_size=init_batch_size, amp=amp
    )

    true_labels = np.array([0] * len(human_scores) + [1] * len(ai_scores))
    prediction_scores = np.array(human_scores + ai_scores, dtype=np.float32)

    auroc = roc_auc_score(true_labels, -prediction_scores)

    return auroc, human_scores, ai_scores

def main():
    device = "cuda:1" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    model_id = os.environ.get("LM_MODEL_ID", "gpt2")
    print(f"Loading pre-trained model ({model_id})...")

    tokenizer = GPT2TokenizerFast.from_pretrained(model_id)
    if tokenizer.pad_token is None:  # pad_token
        tokenizer.pad_token = tokenizer.eos_token

    model = GPT2LMHeadModel.from_pretrained(model_id)
    model.eval().to(device)

    amp = device.startswith("cuda:1")

    dataset_file = "./data/imdb_gemini25flash.json"
    auroc, human_scores, ai_scores = evaluate_and_get_auroc_entropy(
        dataset_file, model, tokenizer, device,
        max_length=1024, init_batch_size=64, amp=amp
    )

    if auroc is not None:
        print("\n" + "="*50)
        print(" " * 12 + "Classification Results (Entropy-based)")
        print("="*50)
        print(f"\nAverage Entropy for HUMAN texts: {np.mean(human_scores):.4f}")
        print(f"Average Entropy for AI texts:    {np.mean(ai_scores):.4f}\n")
        print(f"AUROC Score: {auroc:.4f}")
        print("="*50)


if __name__ == "__main__":
    main()