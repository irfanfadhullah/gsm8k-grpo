from setuptools import setup, find_packages

setup(
    name="gsm8k_grpo",
    version="0.1.0",
    packages=find_packages(include=["config", "data", "models", "training", "evaluation", "utils"]),
    install_requires=[
        "torch>=2.0.0",
        "transformers>=4.30.0",
        "datasets>=2.12.0",
        "peft>=0.4.0",
        "trl>=0.7.1",
        "accelerate>=0.20.0",
    ],
    author="Muhamad Irfan Fadhullah",
    author_email="irfanfadhullah@gmail.com",
    description="Train language models on GSM8K using GRPO",
    keywords="nlp, machine learning, math reasoning, reinforcement learning",
    url="https://github.com/irfanfadhullah/gsm8k-grpo",
    python_requires=">=3.8",
)
