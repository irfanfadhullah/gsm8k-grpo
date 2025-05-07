"""Dataset loading and processing for GSM8K."""
from datasets import load_dataset, Dataset
from typing import Optional

def extract_xml_answer(text: str) -> str:
    """Extract the answer between <answer> tags."""
    if "<answer>" not in text or "</answer>" not in text:
        return ""
    answer = text.split("<answer>")[-1]
    answer = answer.split("</answer>")[0]
    return answer.strip()

def extract_hash_answer(text: str) -> str:
    """Extract answer from GSM8K format (after the #### marker)."""
    if "####" not in text:
        return ""
    return text.split("####")[1].strip().replace(",", "").replace("$", "")


def get_gsm8k_dataset(split="train", max_samples=None, system_prompt="") -> Dataset:
    """Load and format GSM8K dataset for GRPO training."""
    data = load_dataset('openai/gsm8k', 'main')[split]
    
    if max_samples and max_samples < len(data):
        data = data.select(range(max_samples))

    # Format data with prompt structure
    data = data.map(lambda x: {
        'prompt': [
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': x['question']}
        ],
        'answer': extract_hash_answer(x['answer'])
    })

    return data
