"""Reward functions for GRPO training."""
from typing import List, Any
import re
from data.dataset import extract_xml_answer


def correctness_reward_func(prompts, completions, answer, **kwargs) -> List[float]:
    """Reward model if extracted answer matches ground truth (2.0 for correct, 0.0 otherwise)."""
    responses = [completion[0]['content'] for completion in completions]
    extracted_responses = [extract_xml_answer(r) for r in responses]
    
    if kwargs.get('debug', False) and len(responses) > 0:
        print(f"Question: {prompts[0][-1]['content']}")
        print(f"Ground Truth: {answer[0]}")
        print(f"Model Response: {responses[0]}")
        print(f"Extracted Answer: {extracted_responses[0]}")

    return [2.0 if r == a else 0.0 for r, a in zip(extracted_responses, answer)]

def int_reward_func(completions, **kwargs) -> List[float]:
    """Reward model if extracted answer is a digit (0.5 for integer, 0.0 otherwise)."""
    responses = [completion[0]['content'] for completion in completions]
    extracted_responses = [extract_xml_answer(r) for r in responses]
    return [0.5 if r.isdigit() else 0.0 for r in extracted_responses]


def correctness_reward_func(prompts, completions, answer, **kwargs) -> list[float]:
    """
    Reward function that checks if the extracted answer matches the ground truth.
    Returns 2.0 for correct answers, 0.0 otherwise.
    """

    responses = [completion[0]['content'] for completion in completions]
    q = prompts[0][-1]['content']
    extracted_responses = [extract_xml_answer(r) for r in responses]

    # Print debugging info for the first example
    if kwargs.get('debug', False) and len(responses) > 0:
        print('-'*20)
        print(f"Question:\n{q}")
        print(f"\nGround Truth Answer:\n{answer[0]}")
        print(f"\nModel Response:\n{responses[0]}")
        print(f"\nExtracted Answer:\n{extracted_responses[0]}")

    return [2.0 if r == a else 0.0 for r, a in zip(extracted_responses, answer)]

def int_reward_func(completions, **kwargs) -> list[float]:
    """
    Reward function that checks if the extracted answer is a digit.
    Returns 0.5 for integer answers, 0.0 otherwise.
    """

    responses = [completion[0]['content'] for completion in completions]
    extracted_responses = [extract_xml_answer(r) for r in responses]
    return [0.5 if r.isdigit() else 0.0 for r in extracted_responses]

def strict_format_reward_func(completions, **kwargs) -> list[float]:
    """
    Reward function that checks if the completion follows the exact format.
    Returns 0.5 for matching format, 0.0 otherwise.
    """
    
    pattern = r"^\n.*?\n\n\n.*?\n$"
    responses = [completion[0]["content"] for completion in completions]
    matches = [bool(re.search(pattern, r, flags=re.DOTALL)) for r in responses]

    return [0.5 if match else 0.0 for match in matches]

def soft_format_reward_func(completions, **kwargs) -> list[float]:
    """
    Reward function with more lenient format checking.
    Returns 0.5 for matching format, 0.0 otherwise.
    """

    pattern = r".*?\s*.*?"
    responses = [completion[0]["content"] for completion in completions]
    matches = [bool(re.search(pattern, r, flags=re.DOTALL)) for r in responses]

    return [0.5 if match else 0.0 for match in matches]

def count_xml(text) -> float:
    """
    Count XML tags and provide partial rewards for each correctly placed tag.
    """

    count = 0.0
    if text.count("") == 1:
        count += 0.125
    if text.count("") == 1:
        count += 0.125
    if text.count("") == 1:
        count += 0.125
    if text.count("") == 1:
        count += 0.125
        
    return count

def xmlcount_reward_func(completions, **kwargs) -> list[float]:
    """
    Reward function based on counting XML tags in the response.
    """

    contents = [completion[0]["content"] for completion in completions]
    return [count_xml(c) for c in contents]