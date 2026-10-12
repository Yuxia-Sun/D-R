import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, f1_score, roc_auc_score, 
                             roc_curve, confusion_matrix)
import argparse
import warnings
import xgboost as xgb
import os
import numpy as np

warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

def advanced_roc_plot(fpr, tpr, auroc, output_dir):
    plt.figure(figsize=(10, 8))
    
    # 主ROC曲线
    plt.subplot(2, 1, 1)
    plt.plot(fpr, tpr, 'b-', linewidth=2, label=f'XGBoost (AUROC = {auroc:.4f})')
    plt.plot([0, 1], [0, 1], 'k--', label='Random Guessing')
    plt.title('ROC', fontsize=18)
    plt.xlabel('False Positive Rate', fontsize=14)
    plt.ylabel('True Positive Rate', fontsize=14)
    plt.legend(loc='lower right')
    plt.grid(True)
    
    # 放大低FPR区域（0-0.1）
    plt.subplot(2, 1, 2)
    plt.plot(fpr, tpr, 'b-', linewidth=2)
    plt.plot([0, 0.1], [0, 1], 'k--')
    plt.xlim([-0.005, 0.1])
    plt.ylim([0.9, 1.005])
    plt.title('Low FPR Region Zoomed In', fontsize=14)
    plt.xlabel('False Positive Rate (0-0.1)', fontsize=12)
    plt.ylabel('True Positive Rate (0.9-1.0)', fontsize=12)
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'detailed_roc_curve.png'))
    plt.close()

def train_and_evaluate_models(filepath: str, output_dir: str):

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    print(f"data: {filepath} ---")
    
    try:
        df = pd.read_json(filepath)
        df_flat = pd.json_normalize(df.to_dict('records'))
    except FileNotFoundError:
        print(f"not found: '{filepath}'")
        return

    print("\n--- 1.1 ---")
    df_flat = df_flat.sample(frac=1, random_state=11).reset_index(drop=True)
    
    features = [
        'rtr_scores.bert_score_f1', 'rtr_scores.bert_score_p', 'rtr_scores.bert_score_r',
        'rtr_scores.kendall_tau_b', 'rtr_scores.spearman_rho',
        'complexity_scores.flesch_kincaid_grade',
        'complexity_scores.avg_sentence_length',
        'complexity_scores.type_token_ratio'
    ]
    
    features = [f for f in features if f in df_flat.columns]
    print(f" {features}")

    df_flat['label'] = df_flat['source'].apply(lambda x: 1 if x == 'AI' else 0)
    
    X = df_flat[features]
    y = df_flat['label']
    
    initial_count = len(df_flat)
    df_flat = df_flat.dropna(subset=features)
    y = df_flat['label']
    X = df_flat[features]
    
    if initial_count != len(df_flat):
        print(f"移除 {initial_count - len(df_flat)} 个包含缺失值的样本")
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = xgb.XGBClassifier(
        objective='binary:logistic',
        random_state=42, 
        eval_metric='auc',
        learning_rate=0.03,
        n_estimators=400,
        max_depth=3,
        reg_alpha=0.2,
        reg_lambda=1.5
    ) 

    print(f"\n--- 正在训练: XGBoost ---")
    model.fit(X_train_scaled, y_train)
    
    y_pred = model.predict(X_test_scaled)
    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
    
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auroc = roc_auc_score(y_test, y_pred_proba)
    
    print(f"AUROC: {auroc:.4f}, Accuracy: {accuracy:.4f}, F1-Score: {f1:.4f}")

    cv_scores = cross_val_score(
        model, 
        scaler.fit_transform(X), 
        y, 
        cv=5, 
        scoring='roc_auc'
    )
    print(f"5 AUROC: {cv_scores.mean():.4f} 标准差={cv_scores.std():.4f}")


    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    advanced_roc_plot(fpr, tpr, auroc, output_dir)
    
    cm_filename = os.path.join(output_dir, 'confusion_matrix_xgboost.png')
    plt.figure(figsize=(6, 5))
    sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues',
                xticklabels=['Human', 'AI'], yticklabels=['Human', 'AI'])
    plt.title(f'Confusion Matrix', fontsize=16)
    plt.xlabel('Pre-Lable', fontsize=12)
    plt.ylabel('True-Lable', fontsize=12)
    plt.savefig(cm_filename)
    plt.close()

    feature_weights = pd.DataFrame(
        model.feature_importances_, 
        index=features, 
        columns=['Importance']
    ).sort_values(by='Importance', ascending=False)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Importance', y=feature_weights.index, data=feature_weights)
    plt.title("Features' importance", fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'feature_importance.png'))
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.hist(y_pred_proba[y_test == 0], bins=50, alpha=0.5, label='Human')
    plt.hist(y_pred_proba[y_test == 1], bins=50, alpha=0.5, label='AI')
    plt.axvline(0.5, color='red', linestyle='--', label='Threshold = 0.5')
    plt.title('Probability Distribution', fontsize=16)
    plt.xlabel('Predicted Probability', fontsize=14)
    plt.ylabel('Frequency', fontsize=14)
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'probability_distribution.png'))
    plt.close()

    report = f"""
    === 模型评估报告 ===
    数据集: {filepath}
    样本总数: {len(df_flat)}
    AI样本比例: {y.mean():.2%}
    
    主要指标:
    - 测试集 AUROC: {auroc:.6f}
    - 5折交叉验证 AUROC: {cv_scores.mean():.6f} (±{cv_scores.std():.6f})
    - 准确率: {accuracy:.4f}
    - F1分数: {f1:.4f}
    
    特征重要性 (前3):
    {feature_weights.head(3).to_string()}
    """
    
    print(report)
    with open(os.path.join(output_dir, 'model_report.txt'), 'w') as f:
        f.write(report)

    print("\n=== 所有结果已保存到:", output_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and evaluate XGBoost classifier.")
    parser.add_argument("--filepath", type=str, required=True, help="Path to the input JSON file.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save result plots.")
    args = parser.parse_args()

    # qwen
    train_and_evaluate_models(filepath=args.filepath, output_dir=args.output_dir)
    # train_and_evaluate_models('./qwen/cnndaily_article/cnn_article_qwenturbo_result.json',
    #                           './qwen/cnndaily_article/ken_output2')
    # train_and_evaluate_models('./qwen/imdb/imdb_qwenturbo_dsv3_result.json',
    #                           './qwen/imdb/ken_output2')
    # train_and_evaluate_models('./qwen/ml-arxiv/ML-ArXiv-Papers_qwenturbo_dsv3_result.json',
    #                           './qwen/ml-arxiv/ken_output2')
    # train_and_evaluate_models('./qwen/reddit/reddit-title-body_qwenturbo_dsv3.json',
    #                           './qwen/reddit/ken_output2')
    # train_and_evaluate_models('./qwen/ROCStories/ROCStories_qwenturbo_dsv3_result.json',
    #                           './qwen/ROCStories/ken_output2')
    # train_and_evaluate_models('./qwen/wikihow/wikihow_qwenturbo_dsv3_result.json',
    #                           './qwen/wikihow/ken_output2')
    # # gemini
    # train_and_evaluate_models('./Gemini/agnews/ag_news_gemini25flash_dsv3_result.json',
    #                           './Gemini/agnews/ken_output2')
    # train_and_evaluate_models('./Gemini/cnn_article/cnn_article_gemini25flash_dsv3_result.json',
    #                           './Gemini/cnn_article/ken_output2')
    # train_and_evaluate_models('./Gemini/imdb/imdb_gemini25flash_dsv3_result.json',
    #                           './Gemini/imdb/ken_output2')
    # train_and_evaluate_models('./Gemini/ML-ArXiv/ML-ArXiv-Papers_gemini25flash_dsv3_result.json',
    #                           './Gemini/ML-ArXiv/ken_output2')
    # train_and_evaluate_models('./Gemini/reddit/reddit-title-body_gemini25flash_dsv3_result.json',
    #                           './Gemini/reddit/ken_output2')
    # train_and_evaluate_models('./Gemini/ROCStories/ROCStories_gemini25flash_dsv3_result.json',
    #                           './Gemini/ROCStories/ken_output2')
    # train_and_evaluate_models('./Gemini/wikihow/wikihow_gemini25flash_dsv3_result.json',
    #                           './Gemini/wikihow/ken_output2')
    # # gpt
    # train_and_evaluate_models('./gpt/agnews/ag_news_gpt41_dsv3_result.json',
    #                           './gpt/agnews/ken_output2')
    # train_and_evaluate_models('./gpt/cnn_article/cnn_article_openaigpt41nano_dsv3_result.json',
    #                           './gpt/cnn_article/ken_output2')
    # train_and_evaluate_models('./gpt/imdb/imdb_gpt41_dsv3_result.json',
    #                           './gpt/imdb/ken_output2')
    # train_and_evaluate_models('./gpt/ML-ArXiv/ML-ArXiv-Papers_gpt41_dsv3_result.json',
    #                           './gpt/ML-ArXiv/ken_output2')
    # train_and_evaluate_models('./gpt/reddit/reddit-title-body_gpt41_dsv3_result.json',
    #                           './gpt/reddit/ken_output2')
    # train_and_evaluate_models('./gpt/ROCStories/ROCStories_gpt41_dsv3_result.json',
    #                           './gpt/ROCStories/ken_output2')
    # train_and_evaluate_models('./gpt/wikihow/wikihow_gpt41_dsv3_result.json',
    #                           './gpt/wikihow/ken_output2')
    # # grok
    # train_and_evaluate_models('./grok/agnews/ag_news_xaigrok3_dsv3_results.json',
    #                           './grok/agnews/ken_output2')
    # train_and_evaluate_models('./grok/CNNarticle/cnn_dailymail_xaigrok3_dsv3_result.json',
    #                           './grok/CNNarticle/ken_output2')
    # train_and_evaluate_models('./grok/imdb/imdb_xaigrok3_dsv3_result.json',
    #                           './grok/imdb/ken_output2')
    # train_and_evaluate_models('./grok/ML-Arxiv/ML-ArXiv-Papers_grok4_dsv3_result.json',
    #                           './grok/ML-Arxiv/ken_output2')
    # train_and_evaluate_models('./grok/reddit/reddit-title-body_xaigrok3_dsv3_result.json',
    #                           './grok/reddit/ken_output2')
    # train_and_evaluate_models('./grok/ROC/ROCStories_xaigrok3_dsv3_result.json',
    #                           './grok/ROC/ken_output2')
    # train_and_evaluate_models('./grok/wikihow/wikihow_xaigrok3_dsv3_result.json',
    #                           './grok/wikihow/ken_output2')
    # # ds
    # train_and_evaluate_models('./deepseekv3/ag_news/ag_news_deepseek_dsv3_result.json',
    #                           './deepseekv3/ag_news/ken_output2')
    # train_and_evaluate_models('./deepseekv3/cnn_article/cnn_article_deepseek_dsv3_result.json',
    #                           './deepseekv3/cnn_article/ken_output2')
    # train_and_evaluate_models('./deepseekv3/imdb/imdb_deepseek_dsv3_result.json',
    #                           './deepseekv3/imdb/ken_output2')
    # train_and_evaluate_models('./deepseekv3/ML-ArXiv/ML-ArXiv-Papers_deepseek_dsv3_result.json',
    #                           './deepseekv3/ML-ArXiv/ken_output2')
    # train_and_evaluate_models('./deepseekv3/reddit/reddit-title-body_deepseek_dsv3_result.json',
    #                           './deepseekv3/reddit/ken_output2')
    # train_and_evaluate_models('./deepseekv3/ROCStories/ROCStories_deepseek_dsv3_result.json',
    #                           './deepseekv3/ROCStories/ken_output2')
    # train_and_evaluate_models('./deepseekv3/wikihow/wikihow_deepseek_dsv3_result.json',
    #                           './deepseekv3/wikihow/ken_output2')