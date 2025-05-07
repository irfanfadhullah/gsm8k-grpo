#!/usr/bin/env python
"""
Example script to train a model on GSM8K using GRPO.
"""

import argparse
from config import TrainingConfig
from training import GRPOModelTrainer


def main():
    """Parse arguments and run training."""
    parser = argparse.ArgumentParser(description="Train model on GSM8K using GRPO")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-1.5B-Instruct",
                        help="Model to fine-tune")
    parser.add_argument("--output_dir", type=str, default="outputs/grpo-model",
                        help="Directory to save the model")
    parser.add_argument("--learning_rate", type=float, default=5e-5,
                        help="Learning rate")
    parser.add_argument("--epochs", type=int, default=3,
                        help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Batch size")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Maximum number of samples to use")
    
    args = parser.parse_args()
    
    # Create configuration
    config = TrainingConfig(
        model_name=args.model,
        output_dir=args.output_dir,
        learning_rate=args.learning_rate,
        epochs=args.epochs,
        batch_size=args.batch_size,
        max_samples=args.max_samples
    )
    
    # Initialize and run training
    trainer = GRPOModelTrainer(config)
    trainer.train()
    trainer.save_model()


if __name__ == "__main__":
    main()
