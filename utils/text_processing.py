"""Text processing utilities for extracting answers from model responses."""

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
