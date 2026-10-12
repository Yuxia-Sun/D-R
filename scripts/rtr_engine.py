import requests
import torch
import re
import numpy as np
from openai import OpenAI
import textstat
from scipy.stats import kendalltau, spearmanr
from bert_score import score as bert_score_calculate
from collections import Counter

device = "cuda" if torch.cuda.is_available() else "cpu"
device = "mps" if torch.backends.mps.is_available() else "cpu"

print(f"RtR Engine: 将使用 '{device}' 设备进行评估计算。")


def shuffle_text_sentence_level(text: str, seed: int | None = None) -> str:
    import random as _rnd
    rng = _rnd.Random(seed)
    sentences = re.split(r'(?<=[.?!])\s+', text)
    shuffled_sentences = []
    for sentence in sentences:
        if not sentence.strip():
            continue
        words = sentence.split()
        punctuation = ''
        if words and words[-1][-1] in '.?!':
            punctuation = words[-1][-1]
            words[-1] = words[-1][:-1]
        rng.shuffle(words)
        if words:
            shuffled_sentences.append(" ".join(words) + punctuation)
        elif punctuation:
            shuffled_sentences.append(punctuation)
    return " ".join(shuffled_sentences)

'''
from transformers import AutoTokenizer, AutoModelForCausalLM

# 全局模型和tokenizer实例
_tokenizer = None
_model = None

def initialize_model():
    global _tokenizer, _model
    
    if _model is not None and _tokenizer is not None:
        return  # 已经初始化
    
    print("Loading model and tokenizer...wait a moment.")
    
    try:
        MODEL_NAME = "/root/autodl-tmp/R2R/rtr/mistral"
        
        # 加载tokenizer
        _tokenizer = AutoTokenizer.from_pretrained(
            MODEL_NAME, 
            local_files_only=True
        )
        
        # 加载模型
        _model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.bfloat16,
            device_map=device,
            local_files_only=True,
            attn_implementation="flash_attention_2"
        )
        
        print("模型加载成功！")
        
    except Exception as e:
        print(f"模型加载失败: {e}")
        raise RuntimeError(f"无法加载模型: {str(e)}")

def recover_text(shuffled_text: str) -> str:
    global _tokenizer, _model
    
    # 确保模型已加载
    if _model is None or _tokenizer is None:
        initialize_model()
    
    try:
        # 构建系统提示
        system_prompt = (
            "You are an expert in linguistics and text reconstruction. Your task is to unscramble the words to form a coherent, grammatically correct, and meaningful text."
        )
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Please reconstruct the original text from the following scrambled words. Preserve the original vocabulary as much as possible and do not add new information. Return only the reconstructed text, without any introductory phrases.\n\nScrambled words: \"{shuffled_text}\"\n\nReconstructed text:"}
        ]
        
        # 应用聊天模板（不包含tools参数）
        inputs = _tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,  # 添加生成提示符
            return_dict=True,            # 返回字典格式
            return_tensors="pt"           # 返回PyTorch张量
        )
        
        # 移动到模型所在设备
        inputs = {k: v.to(_model.device) for k, v in inputs.items()}
        
        # 生成回复（使用字典解包）
        output = _model.generate(
            **inputs,
            max_new_tokens=512,
            temperature=0.1,
            do_sample=True,
            pad_token_id=_tokenizer.eos_token_id,
            repetition_penalty=1.1
        )
        
        # 提取生成部分（跳过输入提示）
        generated_tokens = output[0][inputs["input_ids"].shape[1]:]
        response = _tokenizer.decode(
            generated_tokens, 
            skip_special_tokens=True
        )
        
        # 清理特殊标记
        return response.strip().replace("<|im_end|>", "").replace("</s>", "")
    
    except Exception as e:
        return f"文本恢复失败: {str(e)}"
'''
# 2. 恢复：只控制 temperature；不设置 top_p（让其保持默认=1）
import os
from openai import OpenAI

def recover_text(
    shuffled_text: str,
    temperature: float = 0.0,
    max_tokens: int = 512,
    seed: int | None = None,          # 有的服务支持，有的不支持；支持就传，不支持会忽略
    model: str | None = None
) -> str:
    try:
        api_key = ''

        base_url = 'https://ark.cn-beijing.volces.com/api/v3'
        # 'https://api.deepseek.com'
        # "https://api.siliconflow.cn/v1"
        # 'https://ark.cn-beijing.volces.com/api/v3'
        api_key  = api_key  or os.getenv("ARK_API_KEY") or os.getenv("OPENAI_API_KEY")
        base_url = base_url or os.getenv("ARK_BASE_URL", "https://ark.cn-beijing.volces.com/api/v3")
        model    = model    or os.getenv("RTR_MODEL_NAME", "deepseek-v3-250324")
        client = OpenAI(api_key=api_key, base_url=base_url)

        messages = [
            {"role": "system",
             "content": "You are an expert in linguistics and text reconstruction. Your task is to unscramble the words to form a coherent, grammatically correct, and meaningful text."},
            {"role": "user",
             "content": f"Please reconstruct the original text from the following scrambled words. Preserve the original vocabulary as much as possible and do not add new information. Return only the reconstructed text.\n\nScrambled words: \"{shuffled_text}\"\n\nReconstructed text:"}
        ]

        params = {
            "model": model,
            "messages": messages,
            "temperature": float(temperature),  # 单因子
            "max_tokens": int(max_tokens),
        }
        if seed is not None:
            params["seed"] = int(seed)          # 若不支持会被忽略

        resp = client.chat.completions.create(**params)
        return resp.choices[0].message.content.strip()
    except Exception as e:
        print(f"调用API时发生异常: {e}")
        return f"错误：API调用失败 - {e}"

def calculate_text_complexity(text: str) -> dict:
    try:
        words = text.split()
        num_words = len(words)
        if num_words == 0:
            return {"flesch_kincaid_grade": 0, "avg_sentence_length": 0, "type_token_ratio": 0}
        
        fk_grade = textstat.flesch_kincaid_grade(text)
        sentences = [s for s in re.split(r'[.?!]', text) if s]
        num_sentences = len(sentences) if len(sentences) > 0 else 1
        avg_sent_len = num_words / num_sentences
        ttr = len(set(words)) / num_words
        
        return {"flesch_kincaid_grade": fk_grade, "avg_sentence_length": avg_sent_len, "type_token_ratio": ttr}
    except Exception as e:
        print(f"计算文本复杂度时出错: {e}")
        return {"flesch_kincaid_grade": -1, "avg_sentence_length": -1, "type_token_ratio": -1}

def calculate_rtr_scores(original_text: str, recovered_text: str) -> dict:
    if not original_text or not recovered_text or "错误" in recovered_text:
        return {"bert_score_p": 0.0, "bert_score_r": 0.0, "bert_score_f1": 0.0, "kendall_tau_b": 0.0, "spearman_rho": 0.0}

    P, R, F1 = bert_score_calculate(
        [recovered_text],
        [original_text],
        lang="en",
        model_type="bert-base-uncased",
        device=device
    )
    
    original_tokens = original_text.lower().split()
    recovered_tokens = recovered_text.lower().split()
    
    kendall_tau, spearman_rho = 0.0, 0.0
    if original_tokens and recovered_tokens:
        original_positions = {}
        counts = Counter()
        for i, token in enumerate(original_tokens):
            original_positions[(token, counts[token])] = i
            counts[token] += 1

        common_tokens_with_counts = []
        recovered_counts = Counter()
        for token in recovered_tokens:
            token_id = (token, recovered_counts[token])
            if token_id in original_positions:
                common_tokens_with_counts.append(token_id)
            recovered_counts[token] += 1
            
        if len(common_tokens_with_counts) >= 2:
            original_rank = [original_positions[token_id] for token_id in common_tokens_with_counts]
            ideal_rank = sorted(original_rank)
            
            tau_val, _ = kendalltau(ideal_rank, original_rank, variant='b')
            kendall_tau = tau_val if not np.isnan(tau_val) else 0.0

            rho_val, _ = spearmanr(ideal_rank, original_rank)
            spearman_rho = rho_val if not np.isnan(rho_val) else 0.0

    return {
        "bert_score_p": P.mean().item(),
        "bert_score_r": R.mean().item(),
        "bert_score_f1": F1.mean().item(),
        "kendall_tau_b": kendall_tau,
        "spearman_rho": spearman_rho
    }