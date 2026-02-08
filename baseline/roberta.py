import json
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F


class TextDetectionDataset(Dataset):
    def __init__(self, data_dict, tokenizer, max_length=512):
        self.texts = []
        self.labels = []

        # Human 0  AI 1
        for text in data_dict["Human"]:
            self.texts.append(text)
            self.labels.append(0)

        for text in data_dict["AI"]:
            self.texts.append(text)
            self.labels.append(1)

        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]

        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }


def load_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data


def evaluate_model(data_path, model_name="roberta-base-openai-detector", batch_size=16):

    data = load_data(data_path)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)

    dataset = TextDetectionDataset(data, tokenizer)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()

    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits

            probabilities = F.softmax(logits, dim=-1)
            ai_probs = probabilities[:, 1].cpu().numpy()

            all_predictions.extend(ai_probs)
            all_labels.extend(labels.cpu().numpy())

    auroc = roc_auc_score(all_labels, all_predictions)

    return auroc, all_predictions, all_labels


def main():
    data_path = "xxxxxx.json"

    try:
        auroc, predictions, labels = evaluate_model(data_path)

        print(f"AUROC Score: {auroc:.4f}")
        print(f"Total samples: {len(labels)}")
        print(f"Human samples: {sum(1 for l in labels if l == 0)}")
        print(f"AI samples: {sum(1 for l in labels if l == 1)}")

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
