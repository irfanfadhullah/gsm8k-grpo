"""Model loading utilities."""
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model

def load_model_and_tokenizer(model_name, use_peft=True, peft_config=None, hf_cache_dir=None):
    """Load a pre-trained model and tokenizer with optional PEFT configuration."""
    if hf_cache_dir:
        import os
        os.environ["HF_HOME"] = hf_cache_dir
    
    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        low_cpu_mem_usage=True,
        trust_remote_code=True
    )
    
    # Apply PEFT if requested
    if use_peft and peft_config:
        model = get_peft_model(model, peft_config)
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    return model, tokenizer
