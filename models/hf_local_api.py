from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import os

MODEL_CACHE = {}

MODEL_MAPPING = {
    "gpt-neo-1.3B": "EleutherAI/gpt-neo-1.3B",
    "gpt-neo-2.7B": "EleutherAI/gpt-neo-2.7B",
    "chatglm3": "THUDM/chatglm3-6b",
    "deepseek": "deepseek-ai/deepseek-llm-7b-chat"
}

def call_model(prompt: str, model_name: str, max_new_tokens: int = 75) -> str:
    assert model_name in MODEL_MAPPING, f"Unsupported model: {model_name}"

    if model_name not in MODEL_CACHE:
        print(f"[INFO] Loading model {model_name}...")
        model_id = MODEL_MAPPING[model_name]
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32)
        model.eval()
        if torch.cuda.is_available():
            model.to("cuda")
        MODEL_CACHE[model_name] = (tokenizer, model)

    tokenizer, model = MODEL_CACHE[model_name]

    inputs = tokenizer(prompt, return_tensors="pt")
    if torch.cuda.is_available():
        inputs = {k: v.to("cuda") for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=True, temperature=0.7, top_p=0.95)
    generated = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return generated[len(prompt):].strip() if generated.startswith(prompt) else generated

if __name__ == '__main__':
    print(call_model("What is your name?", "gpt-neo-1.3B"))