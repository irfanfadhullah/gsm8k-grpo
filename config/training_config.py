"""Configuration for GRPO training."""
from dataclasses import dataclass, field
from typing import List, Optional, Union, Dict, Any
from peft import LoraConfig


@dataclass
class TrainingConfig:
    """Configuration for GRPO training."""
    
    # Model configuration
    model_name: str = "Qwen/Qwen2.5-1.5B-Instruct"
    output_dir: str = "outputs/grpo-model"
    
    # Training parameters
    learning_rate: float = 5e-5
    epochs: int = 3
    batch_size: int = 8
    max_samples: Optional[int] = None
    gradient_accumulation_steps: int = 4
    
    # PEFT configuration
    use_peft: bool = True
    peft_config: Optional[LoraConfig] = field(default_factory=lambda: LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]
    ))
    
    # Dataset configuration
    dataset_name: str = "openai/gsm8k"
    dataset_split: str = "train"
    
    # GRPO parameters
    reward_functions: List[str] = field(default_factory=lambda: ["correctness", "integer"])
    reward_weights: List[float] = field(default_factory=lambda: [0.8, 0.2])
    kl_coef: float = 0.1
    
    # Prompt configuration
    system_prompt: str = (
        "You are a helpful math assistant. Solve the problem step-by-step "
        "and give your final numerical answer between <answer></answer> tags."
    )
    
    # Logging configuration
    use_wandb: bool = False
    wandb_project: str = "gsm8k-grpo"
    logging_steps: int = 10
    eval_steps: int = 100
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        if len(self.reward_functions) != len(self.reward_weights):
            raise ValueError("Number of reward functions must match number of weights")
