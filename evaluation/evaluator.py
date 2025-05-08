"""Evaluation utilities for assessing model performance on GSM8K."""

import torch
from tqdm import tqdm
from typing import Dict, List, Any, Callable, Optional
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from utils.text_processing import extract_xml_answer, extract_hash_answer


class ModelEvaluator:
    """Evaluator for assessing model performance on math problems."""
    
    def __init__(
        self, 
        model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
        system_prompt: str = "",
        extract_fn: Callable = extract_xml_answer,
        device: Optional[str] = None
    ):
        """Initialize evaluator with model and tokenizer.
        
        Args:
            model: The model to evaluate
            tokenizer: The tokenizer for the model
            system_prompt: System prompt to prepend to user queries
            extract_fn: Function to extract the final answer from model output
            device: Device to run evaluation on (defaults to model's device)
        """
        self.model = model
        self.tokenizer = tokenizer
        self.system_prompt = system_prompt
        self.extract_fn = extract_fn
        self.device = device or next(model.parameters()).device
    
    def evaluate_dataset(
        self, 
        dataset: Dataset,
        batch_size: int = 4,
        max_new_tokens: int = 512
    ) -> Dict[str, float]:
        """Evaluate model on a dataset of math problems.
        
        Args:
            dataset: Dataset containing prompts and answers
            batch_size: Batch size for evaluation
            max_new_tokens: Maximum number of tokens to generate
            
        Returns:
            Dictionary with metrics including accuracy
        """
        correct_count = 0
        total_count = 0
        
        # Process in batches
        for i in tqdm(range(0, len(dataset), batch_size)):
            batch = [dataset[j] for j in range(i, min(i+batch_size, len(dataset)))]

            # batch = dataset[i:min(i+batch_size, len(dataset))]
            
            # Format inputs
            # prompts = [
            #     [
            #         {"role": "system", "content": self.system_prompt},
            #         {"role": "user", "content": item["question"]}
            #     ] 
            #     for item in batch
            # ]
            prompts = [
                item["prompt"]  # This already contains the system prompt and user question
                for item in batch
            ]
            
            # Generate responses
            responses = self._generate_responses(prompts, max_new_tokens)
            extracted_answers = [self.extract_fn(r) for r in responses]
            
            # Check correctness
            for idx, extracted_answer in enumerate(extracted_answers):
                gold_answer = extract_hash_answer(batch[idx]["answer"])
                
                # Try to normalize answers for comparison
                try:
                    extracted_numeric = float(extracted_answer.replace(",", "").replace("$", ""))
                    gold_numeric = float(gold_answer.replace(",", "").replace("$", ""))
                    is_correct = abs(extracted_numeric - gold_numeric) < 1e-6
                except ValueError:
                    is_correct = extracted_answer == gold_answer
                
                if is_correct:
                    correct_count += 1
                total_count += 1
        
        return {
            "accuracy": correct_count / total_count if total_count > 0 else 0,
            "correct": correct_count,
            "total": total_count
        }
    
    def _generate_responses(self, prompts, max_new_tokens):
        """Generate responses for a batch of prompts."""
        responses = []
        
        with torch.no_grad():
            for prompt in prompts:
                inputs = self.tokenizer.apply_chat_template(
                    prompt, 
                    return_tensors="pt"
                ).to(self.device)
                
                output = self.model.generate(
                    inputs, 
                    max_new_tokens=max_new_tokens,
                    do_sample=False
                )
                
                # Decode and extract assistant's response
                full_response = self.tokenizer.decode(output[0], skip_special_tokens=True)
                # Extract just the assistant's part
                assistant_response = full_response.split(prompt[-1]["content"])[-1].strip()
                responses.append(assistant_response)
        
        return responses
