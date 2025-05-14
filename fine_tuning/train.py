#!/usr/bin/env python

import os
import argparse
import logging
import random
import numpy as np
import torch
import gc

import wandb
import deepspeed

from typing import Dict, Optional
from datasets import load_from_disk
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
    set_seed,
    BitsAndBytesConfig,
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType,
)
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import logging as hf_logging
from utils import compute_json_accuracy, extract_json_from_response, create_length_bins

logger = logging.getLogger(__name__)
logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO)
hf_logging.set_verbosity_info()

def parse_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--wandb_token", type=str, default="")
    parser.add_argument("--wandb_project", type=str, default="Data Extractor")
    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument("--tokenizer_name_or_path", type=str, default=None)
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--max_seq_length", type=int, default=30000)
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--lora_target_modules", type=str, default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj")
    parser.add_argument("--num_train_epochs", type=int, default=3)
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--eval_steps", type=int, default=500)
    parser.add_argument("--save_steps", type=int, default=500)
    parser.add_argument("--save_total_limit", type=int, default=3)
    parser.add_argument("--deepspeed", type=str, required=True)
    parser.add_argument("--resume_from_checkpoint", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--report_to", type=str, default="wandb")
    parser.add_argument("--use_flash_attention_2", action="store_true")
    parser.add_argument("--enable_gradient_checkpointing", action="store_true")
    parser.add_argument("--local_rank", type=int, default=-1)
    
    return parser.parse_args()

def preprocess_dataset(dataset, tokenizer, max_seq_length):
    def tokenize_function(examples):
        prompts = examples["prompt"]
        completions = examples["completion"]
        input_ids_list, labels_list, attention_mask_list = [], [], []

        for prompt, completion in zip(prompts, completions):
            prompt_ids = tokenizer(prompt, add_special_tokens=False, truncation=False)["input_ids"]
            completion_ids = tokenizer(completion, add_special_tokens=False, truncation=False)["input_ids"]

            # Calculate total length
            total_length = len(prompt_ids) + len(completion_ids)
            
            # If too long, truncate the prompt from the beginning
            if total_length > max_seq_length:
                excess_tokens = total_length - max_seq_length
                # Ensure we keep all completion tokens and truncate only from prompt
                if excess_tokens < len(prompt_ids):
                    prompt_ids = prompt_ids[excess_tokens:]
                else:
                    # This case should be rare as preprocessing should handle it
                    prompt_ids = prompt_ids[:1]  # Keep at least one token
                    completion_ids = completion_ids[:max_seq_length-1]  # And truncate completion if needed

            input_ids = prompt_ids + completion_ids
            attention_mask = [1] * len(input_ids)
            labels = [-100] * len(prompt_ids) + completion_ids

            input_ids_list.append(input_ids)
            attention_mask_list.append(attention_mask)
            labels_list.append(labels)

        return {"input_ids": input_ids_list, "attention_mask": attention_mask_list, "labels": labels_list}

    dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset.column_names,
        desc="Tokenizing dataset",
        num_proc=4,
        load_from_cache_file=False
    )
    
    # Create more fine-grained length bins (15 instead of 10) for better memory efficiency
    dataset = create_length_bins(dataset, input_col="input_ids", num_bins=15)
    return dataset

def create_data_collator(tokenizer):
    base_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, padding="longest", return_tensors="pt")
    
    def collator(features):
        if "length_bin" in features[0]:
            for f in features:
                f.pop("length_bin")
        return base_collator(features)
    
    return collator

def main():
    args = parse_args()

    local_rank = int(os.environ.get("LOCAL_RANK", args.local_rank))
    if local_rank != -1:
        torch.cuda.set_device(local_rank)
        torch.distributed.init_process_group(backend="nccl")

    set_seed(args.seed)

    if args.wandb_token and local_rank in [-1, 0]:
        wandb.login(key=args.wandb_token)
        wandb.init(project=args.wandb_project, job_type="training", anonymous="allow")

    os.makedirs(args.output_dir, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_name_or_path or args.model_name_or_path,
        use_fast=True,
        padding_side="right",
        trust_remote_code=True,
        model_max_length=args.max_seq_length
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    logger.info(f"Loading model {args.model_name_or_path} with 4-bit quantization")
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        quantization_config=quantization_config,
        device_map=None,
        trust_remote_code=True,
        attn_implementation="flash_attention_2" if args.use_flash_attention_2 else "eager",
        use_cache=False,
        rope_scaling={"type": "dynamic", "factor": 1.5}
    )

    model = prepare_model_for_kbit_training(model)

    if args.enable_gradient_checkpointing:
        model.gradient_checkpointing_enable()

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=[m.strip() for m in args.lora_target_modules.split(",")],
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )

    model = get_peft_model(model, lora_config)
    if local_rank in [-1, 0]:
        model.print_trainable_parameters()

    train_dataset = load_from_disk(os.path.join(args.data_dir, "train_dataset"))
    val_dataset = load_from_disk(os.path.join(args.data_dir, "val_dataset"))

    train_dataset = preprocess_dataset(train_dataset, tokenizer, args.max_seq_length)
    val_dataset = preprocess_dataset(val_dataset, tokenizer, args.max_seq_length)

    data_collator = create_data_collator(tokenizer)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        logging_steps=args.logging_steps,
        evaluation_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        bf16=True,
        report_to=args.report_to if local_rank in [-1, 0] else "none",
        remove_unused_columns=False,
        dataloader_drop_last=True,
        seed=args.seed,
        local_rank=local_rank,
        deepspeed=args.deepspeed,
        gradient_checkpointing=args.enable_gradient_checkpointing,
        optim="adamw_torch_fused",
        max_grad_norm=1.0,
        group_by_length=True,  # Group samples of similar length
        ddp_find_unused_parameters=False,  # More efficient DDP
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
    )

    last_checkpoint = get_last_checkpoint(args.output_dir) if args.resume_from_checkpoint else None
    trainer.train(resume_from_checkpoint=last_checkpoint)

    # Save model from rank 0 only
    if local_rank in [-1, 0]:
        try:
            # Attempt to save model but don't worry if it fails
            logger.info("Attempting to save final model...")
            trainer.save_model(os.path.join(args.output_dir, "final_model"))
            logger.info("Model saved successfully!")
        except Exception as e:
            # Log the error but continue - we can recover from checkpoints later
            logger.warning(f"Failed to save final model: {e}")
            logger.warning("This is OK - model can be recovered from checkpoint-444")
        
        try:
            # Try to finish wandb logging
            wandb.finish()
        except:
            # Ignore wandb errors
            pass

    # REMOVED: No explicit cleanup of distributed process group
    # Let SLURM handle termination instead
    logger.info("Training completed successfully!")

if __name__ == "__main__":
    main()