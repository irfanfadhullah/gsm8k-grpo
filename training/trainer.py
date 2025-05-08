"""GRPO Trainer for fine-tuning language models on math problems."""

import os
import torch
from typing import List, Dict, Any, Optional, Callable
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import create_reference_model
from trl import GRPOConfig, GRPOTrainer

from config import TrainingConfig
from data import get_gsm8k_dataset
from models import load_model_and_tokenizer
from .reward_functions import correctness_reward_func, int_reward_func, xmlcount_reward_func, soft_format_reward_func


class GRPOModelTrainer:
    """Trainer for fine-tuning models using GRPO."""
    
    def __init__(self, config: TrainingConfig):
        """Initialize trainer with configuration.
        
        Args:
            config: Training configuration
        """
        self.config = config
        self.model = None
        self.tokenizer = None
        self.trainer = None
        
        # Map reward function names to actual functions
        self.reward_fn_map = {
            "correctness": correctness_reward_func,
            "integer": int_reward_func,
        }
        
        # Set up model, tokenizer and dataset
        self._setup()
    
    def _setup(self):
        """Set up model, tokenizer, and dataset."""
        # Load model and tokenizer
        self.model, self.tokenizer = load_model_and_tokenizer(
            self.config.model_name,
            use_peft=self.config.use_peft,
            peft_config=self.config.peft_config
        )
        
        # Create reference model for KL penalty
        # Note: This is still created but handled differently by GRPO
        self.ref_model = create_reference_model(self.model)
        
        # Load dataset
        self.dataset = get_gsm8k_dataset(
            split=self.config.dataset_split,
            max_samples=self.config.max_samples,
            system_prompt=self.config.system_prompt
        )
        
        # Set up reward function (will handle separately in trainer)
        self.reward_fns = [self.reward_fn_map[fn] for fn in self.config.reward_functions]
        
        # Initialize GRPO configuration with supported parameters
        self.grpo_config = GRPOConfig(
            output_dir=self.config.output_dir,
            learning_rate=self.config.learning_rate,
            per_device_train_batch_size=self.config.batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            logging_steps=self.config.logging_steps,
            num_train_epochs=self.config.epochs,
            fp16=True,
            max_prompt_length=256,
            max_completion_length=512,
            num_generations=4,  # Must be divisible by batch_size * grad_accum
            save_steps=100,
            max_grad_norm=0.1,
            report_to="none",
            # Add other necessary parameters from the example
            warmup_ratio=0.1,
            lr_scheduler_type='cosine',
            weight_decay=0.1,
            adam_beta1=0.9,
            adam_beta2=0.99,
            # If KL penalty coefficient is needed, add it here
            # kl_coef=0.1,  # This parameter name might vary based on GRPOConfig implementation
        )
        
        # Initialize GRPO trainer (removed ref_model parameter)
        self.trainer = GRPOTrainer(
            model=self.model,
            processing_class=self.tokenizer,
            train_dataset=self.dataset,
            reward_funcs=[
                xmlcount_reward_func,
                soft_format_reward_func,
                int_reward_func,
                correctness_reward_func
            ],
            peft_config=self.config.peft_config if self.config.use_peft else None,
            args=self.grpo_config
        )
    
    def train(self):
        """Train the model using GRPO."""
        if self.trainer is None:
            raise ValueError("Trainer not initialized. Call _setup() first.")
        
        self.trainer.train()
    
    def save_model(self, path: Optional[str] = None):
        """Save the trained model."""
        output_dir = path or self.config.output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        if self.config.use_peft:
            self.model.save_pretrained(output_dir)
        else:
            self.model.save_pretrained(output_dir)
        
        self.tokenizer.save_pretrained(output_dir)
        
        print(f"Model saved to {output_dir}")