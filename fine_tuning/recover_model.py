#!/usr/bin/env python
"""
Recover a trained LoRA model from a saved checkpoint.
This script can be used when the training completes but the final model save fails.
"""

import os
import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, PeftConfig

def parse_args():
    parser = argparse.ArgumentParser(description="Recover trained LoRA model from checkpoint")
    
    parser.add_argument("--base_model", type=str, required=True, 
                        help="Base model name or path (e.g., meta-llama/Llama-3.2-3B-Instruct)")
    parser.add_argument("--checkpoint_dir", type=str, required=True, 
                        help="Directory containing the checkpoint (e.g., results/checkpoint-1000)")
    parser.add_argument("--output_dir", type=str, required=True, 
                        help="Directory to save the recovered model")
    parser.add_argument("--device", type=str, default="auto", 
                        help="Device to load model on ('cpu', 'cuda', or 'auto')")
    parser.add_argument("--save_merged", action="store_true",
                        help="Save a fully merged model instead of adapter-only")
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    print(f"Loading tokenizer from {args.base_model}")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    
    print(f"Loading base model from {args.base_model}")
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16,
        device_map=args.device
    )
    
    # Verify the checkpoint exists and contains the adapter config
    adapter_config_path = os.path.join(args.checkpoint_dir, "adapter_config.json")
    if not os.path.exists(adapter_config_path):
        raise FileNotFoundError(f"Adapter config not found at {adapter_config_path}")
    
    print(f"Loading adapter from {args.checkpoint_dir}")
    adapter_model = PeftModel.from_pretrained(model, args.checkpoint_dir)
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    if args.save_merged:
        print("Merging adapter weights with base model")
        merged_model = adapter_model.merge_and_unload()
        
        print(f"Saving merged model to {args.output_dir}")
        merged_model.save_pretrained(args.output_dir)
    else:
        print(f"Saving adapter model to {args.output_dir}")
        adapter_model.save_pretrained(args.output_dir)
    
    print(f"Saving tokenizer to {args.output_dir}")
    tokenizer.save_pretrained(args.output_dir)
    
    print("Model recovery completed successfully!")

if __name__ == "__main__":
    main()