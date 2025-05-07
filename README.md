# GSM8K-GRPO

Train language models on mathematical reasoning tasks using Generative Reinforcement Learning from Preference Optimization (GRPO).

## Overview

This repository implements GRPO training for language models to solve math problems from the GSM8K dataset. The training encourages models to:

1. Provide step-by-step reasoning
2. Format answers consistently using XML tags
3. Generate correct numerical answers

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/gsm8k-grpo.git
cd gsm8k-grpo

# Install dependencies
pip install -r requirements.txt

# Install the package in development mode
pip install -e .
```


## Running the Code

### Method 1: Using the Python API

```python
from gsm8k_grpo.config import TrainingConfig
from gsm8k_grpo.training import GRPOModelTrainer

# Create configuration
config = TrainingConfig(
    model_name="Qwen/Qwen2.5-1.5B-Instruct",
    output_dir="outputs/my-model",
    learning_rate=5e-5,
    epochs=3,
    batch_size=8
)

# Initialize and run training
trainer = GRPOModelTrainer(config)
trainer.train()
trainer.save_model()
```


### Method 2: Using the Command-line Script

```bash
# Run with default settings
python scripts/train_model.py

# Run with custom parameters
python scripts/train_model.py \
    --model "Qwen/Qwen2.5-1.5B-Instruct" \
    --output_dir "outputs/custom-model" \
    --learning_rate 3e-5 \
    --epochs 5 \
    --batch_size 4 \
    --max_samples 1000
```


## Configuration Options

| Parameter | Description | Default |
| :-- | :-- | :-- |
| `model_name` | Hugging Face model identifier | "Qwen/Qwen2.5-1.5B-Instruct" |
| `output_dir` | Directory to save model outputs | "outputs/grpo-model" |
| `learning_rate` | Learning rate for optimizer | 5e-5 |
| `epochs` | Number of training epochs | 3 |
| `batch_size` | Training batch size | 8 |
| `max_samples` | Maximum samples to use (None = all) | None |
| `gradient_accumulation_steps` | Gradient accumulation steps | 4 |
| `use_peft` | Whether to use PEFT/LoRA | True |

## Evaluating a Trained Model

```python
from gsm8k_grpo.models import load_model_and_tokenizer
from gsm8k_grpo.evaluation import ModelEvaluator
from gsm8k_grpo.data import get_gsm8k_dataset

# Load model and tokenizer
model, tokenizer = load_model_and_tokenizer("outputs/my-model")

# Create evaluator
evaluator = ModelEvaluator(
    model=model,
    tokenizer=tokenizer,
    system_prompt="You are a helpful math assistant. Solve the problem step-by-step and give your final numerical answer between <answer></answer> tags."
)

# Load evaluation dataset
eval_dataset = get_gsm8k_dataset(split="test", max_samples=100)

# Run evaluation
results = evaluator.evaluate_dataset(eval_dataset)
print(f"Accuracy: {results['accuracy']:.2%}")
```
