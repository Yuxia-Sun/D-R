import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, roc_curve, confusion_matrix, classification_report
import argparse
import warnings

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

import xgboost as xgb
import lightgbm as lgb

warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc



def train_and_evaluate_models(filepath: str):

    try:
        df = pd.read_json(filepath)
        df_flat = pd.json_normalize(df.to_dict('records'))
    except FileNotFoundError:
        return

    df_flat = df_flat.sample(frac=1, random_state=11).reset_index(drop=True)
    
    features = [
    ]
    
    features = [f for f in features if f in df_flat.columns]
    
    df_flat['label'] = df_flat['source'].apply(lambda x: 1 if x == 'AI' else 0)
    
    X = df_flat[features]
    y = df_flat['label']

    if X.isnull().sum().sum() > 0:
        print(X.isnull().sum())
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)


    models = {
        'Logistic Regression': LogisticRegression(random_state=42, class_weight='balanced'),
        'Random Forest': RandomForestClassifier(random_state=42, class_weight='balanced'),
        'XGBoost': xgb.XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss'),
        'LightGBM': lgb.LGBMClassifier(random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(random_state=42),
        'Support Vector Machine': SVC(random_state=42, class_weight='balanced', probability=True),
        'K-Nearest Neighbors': KNeighborsClassifier(),
        'Decision Tree': DecisionTreeClassifier(random_state=42, class_weight='balanced')
    }
    
    results = []

    plt.figure(figsize=(12, 10))
    
    for name, model in models.items():
        print(f"\n---{name}---")
        
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]

        auroc = roc_auc_score(y_test, y_pred_proba)
        results.append({"Model": name, "AUROC": auroc})
        
        print(f"AUROC: {auroc:.4f}")
        
        plt.figure(figsize=(6, 5))
        sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues',
                        xticklabels=['Human', 'AI'], yticklabels=['Human', 'AI'])
        plt.title(f'{name} - confusion', fontsize=16)
        plt.xlabel('lable1', fontsize=12)
        plt.ylabel('lable2', fontsize=12)
        cm_filename = f'confusion_matrix_{name.replace(" ", "_").lower()}.png'
        plt.savefig(cm_filename)
        plt.close()

        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        plt.figure(1) 
        plt.plot(fpr, tpr, label=f'{name} (AUROC = {auroc:.4f})')
        print(f"prob：\n{y_pred_proba[:10]}")

    results_df = pd.DataFrame(results).sort_values(by='AUROC', ascending=False).reset_index(drop=True)

    print(results_df.to_string())

    best_model_name = results_df.iloc[0]['Model']

    best_model = models[best_model_name]
    y_pred_best = best_model.predict(X_test_scaled)
    print(classification_report(y_test, y_pred_best, target_names=['Human', 'AI']))

    if hasattr(best_model, 'feature_importances_'):
        feature_weights = pd.DataFrame(
            best_model.feature_importances_, 
            index=features, 
            columns=['Importance']
        ).sort_values(by='Importance', ascending=False)
        print(feature_weights)
    elif hasattr(best_model, 'coef_'):
        feature_weights = pd.DataFrame(
            best_model.coef_[0], 
            index=features, 
            columns=['Weight']
        ).sort_values(by='Weight', ascending=False)
        print(feature_weights)

    print("\n------")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="find one.")
    parser.add_argument("--filepath", type=str, required=True, help="Path to the input JSON file.")
    args = parser.parse_args()
    
    train_and_evaluate_models(filepath=args.filepath)