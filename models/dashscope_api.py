import os
import dashscope
from http import HTTPStatus
import time

DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY") or "sk-xxxx"

MODEL_MAPPING = {
    "qwen-max": "qwen-max",
    "qwen-plus": "qwen-plus",
    "qwen-turbo": "qwen-turbo",
    "llama3.2": "llama3.2-70b-chat",
}

def call_model(prompt: str, model_name: str, retries: int = 3) -> str:
    assert model_name in MODEL_MAPPING, f"Unsupported model: {model_name}"

    for attempt in range(retries):
        try:
            response = dashscope.Generation.call(
                model=MODEL_MAPPING[model_name],
                api_key=DASHSCOPE_API_KEY,
                prompt=prompt,
                temperature=0.7,
                top_p=0.95,
                max_tokens=75,
                result_format='text'
            )
            if response.status_code == HTTPStatus.OK:
                return response.output['text']
            else:
                print(f"[WARN] API call failed with code {response.status_code}: {response.message}")
        except Exception as e:
            print(f"[ERROR] Exception when calling {model_name}: {e}")
            time.sleep(1.5 * (attempt + 1))

    return f"[ERROR] Failed to generate text from {model_name} after {retries} retries."

if __name__ == '__main__':
    os.environ['DASHSCOPE_API_KEY'] = "sk-xxxx"
    result = call_model("Tell me a story about AI.", "qwen-max")
    print(result)
