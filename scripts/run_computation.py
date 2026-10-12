import os
import json
import pandas as pd
import torch
import time
from rtr_engine import shuffle_text_sentence_level, recover_text, calculate_text_complexity, calculate_rtr_scores #, initialize_model
import argparse

'''
from transformers import AutoTokenizer, AutoModel
BERT_SCORE_MODEL_PATH = "models/bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(BERT_SCORE_MODEL_PATH)
model = AutoModel.from_pretrained(BERT_SCORE_MODEL_PATH).to(device)
'''

def main(input_path: str, output_path: str):
    print("\n--- 开始运行RtR计算流程 ---")  
    # initialize_model()
    # 1. 从本地JSON文件加载数据
    try:
        with open(input_path, 'r', encoding='utf-8') as f:
            samples_to_run = json.load(f)
        print(f"从 '{input_path}' 加载原始数据成功。")
        print(f"将处理 {len(samples_to_run.get('Human', []))} 对样本。")
    except FileNotFoundError:
        print(f"[错误] 输入文件未找到: '{input_path}'")
        print("请先运行 'prepare_dataset.py' 来生成原始数据文件。")
        return

    all_results_data = []
    start_time = time.time()

    for source, texts in samples_to_run.items():
        print(f"\n--- 正在处理来源为 '{source}' 的文本 ---")
        for i, text in enumerate(texts):
            if not text or not isinstance(text, str):
                print(f"跳过无效样本 {i+1} (来源: {source})")
                continue
                
            print(f"--- 处理样本 {i+1}/{len(texts)} (来源: {source}) ---")
            sample_start_time = time.time()
            # 2. 计算原始文本的复杂度
            complexity_scores = calculate_text_complexity(text)
            
            # 3. 扰乱
            shuffled = shuffle_text_sentence_level(text)
            
            # 4. 恢复
            print("正在恢复文本...")
            recovered = recover_text(shuffled)
            
            # 5. 评估恢复质量
            rtr_scores = calculate_rtr_scores(text, recovered)
            
            # 6. 记录所有数据到一个结构中
            sample_result = {
                "id": f"{source}_{i+1}",
                "source": source,
                "original_text": text,
                "shuffled_text": shuffled,
                "recovered_text": recovered,
                "complexity_scores": complexity_scores,
                "rtr_scores": rtr_scores
            }
            all_results_data.append(sample_result)

            print(f"当前样本评估分数: BERT-F1={rtr_scores['bert_score_f1']:.4f}, Kendall-Tau={rtr_scores['kendall_tau_b']:.4f}")

            sample_end_time = time.time()
            sample_duration = sample_end_time - sample_start_time
            total_elapsed = sample_end_time - start_time
            avg_time_per_sample = total_elapsed / len(all_results_data)
            print(
                f"本轮耗时: {sample_duration:.2f}秒，当前总耗时: {total_elapsed:.2f}秒，平均耗时: {avg_time_per_sample:.2f}秒")
    total_time = time.time() - start_time
    avg_time = total_time / len(all_results_data) if all_results_data else 0
    print("\n--- 计算流程结束 ---\n")
    print(f"总处理时间: {total_time:.2f}秒，平均每个样本: {avg_time:.2f}秒")

    if not all_results_data:
        print("没有收集到任何计算结果。")
        return

    # 7. 将所有结果保存到一个JSON文件中
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(all_results_data, f, ensure_ascii=False, indent=2)
    print(f"所有详细计算结果已保存到 {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run RtR computation pipeline.")
    parser.add_argument('--input', type=str, required=True, help='Path to input JSON file')
    parser.add_argument('--output', type=str, required=True, help='Path to output JSON file')
    args = parser.parse_args()

    main(input_path=args.input, output_path=args.output)
