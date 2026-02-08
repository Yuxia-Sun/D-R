import json
from datasets import load_dataset
import argparse

def download_and_save_data(nsamples: int, output_path: str, dataset_name: str = "HC3"):
    print(f"---：{dataset_name}")
    
    try:
        if dataset_name == "HC3":
            dataset = load_dataset("Hello-SimpleAI/HC3", "all", split=f"train[:{nsamples}]", trust_remote_code=True)
            human_samples = [item['human_answers'][0] for item in dataset if item['human_answers'] and item['human_answers'][0]]
            chatgpt_samples = [item['chatgpt_answers'][0] for item in dataset if item['chatgpt_answers'] and item['chatgpt_answers'][0]]            
        elif dataset_name == "custom":
            with open(f"{dataset_name}_data.json", 'r', encoding='utf-8') as f:
                custom_data = json.load(f)
            human_samples = custom_data.get('human', [])
            chatgpt_samples = custom_data.get('ai', [])
            
        else:
            raise ValueError(f"error: {dataset_name}")
        
        source_data = {
            "Human": human_samples,
            "AI": chatgpt_samples
        }

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(source_data, f, ensure_ascii=False, indent=2)


    except Exception as e:
        print(f"error: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download and save dataset samples.")
    parser.add_argument("--nsamples", type=int, default=1000, help="Number of samples to process.")
    parser.add_argument("--output_path", type=str, default="source_data.json", help="Path to save the JSON data.")
    parser.add_argument("--dataset", type=str, default="HC3", choices=["HC3", "M4", "TuringBench", "custom"], help="Dataset to use.")
    args = parser.parse_args()

    download_and_save_data(nsamples=args.nsamples, output_path=args.output_path, dataset_name=args.dataset)