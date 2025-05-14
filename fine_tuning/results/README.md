# Fine-Tuning Results Directory

This directory contains the results of fine-tuning the MetaMine model, including checkpoints and the final fine-tuned model.

## Directory Structure

- **checkpoint-444/**: Checkpoint from the fine-tuning process
  - Contains model weights, training state, and configuration files for resuming training
  - Includes adapter model weights in the PEFT (Parameter-Efficient Fine-Tuning) format
  - RNG states for reproducibility
  - Training arguments and scheduler state
  - Zero-to-FP32 conversion script for DeepSpeed

- **final_model/**: The complete fine-tuned model ready for inference
  - **adapter_model.safetensors**: The LoRA adapter weights (~97 MB)
  - **adapter_config.json**: Configuration for the LoRA adapter
  - **tokenizer.json**: Tokenizer model file
  - **tokenizer_config.json**: Tokenizer configuration
  - **special_tokens_map.json**: Mapping of special tokens
  - **README.md**: Model card template

- **ds_config.json**: DeepSpeed configuration file used for training
  - Contains settings for ZeRO-3 optimization
  - BF16 mixed precision training
  - Gradient accumulation steps (32)
  - Micro-batch size (1 per GPU)

## Model Architecture

The fine-tuned model is based on the Llama-3.2-3B-Instruct model with LoRA adapters. The training applied the following:

- **Base model**: meta-llama/Llama-3.2-3B-Instruct (3 billion parameters)
- **Fine-tuning method**: LoRA (Low-Rank Adaptation)
- **LoRA parameters**:
  - Rank: 16
  - Alpha: 32
  - Dropout: 0.05
  - Target modules: Attention and MLP projection matrices (q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj)

## Training Configuration

The model was trained using:

- **Hardware**: 4 NVIDIA H100 GPUs
- **Framework**: Hugging Face Transformers with DeepSpeed ZeRO-3
- **Precision**: BF16 mixed precision
- **Batch size**: 1 per GPU
- **Gradient accumulation steps**: 32
- **Context length**: 30,000 tokens
- **Optimizer**: AdamW with fused implementation
- **Learning rate**: 2e-4
- **Weight decay**: 0.01
- **Warmup ratio**: 0.1
- **Training time**: Approximately 9 hours

## Using the Model

To load the fine-tuned model for inference:

```python
from peft import PeftModel, PeftConfig
from transformers import AutoModelForCausalLM, AutoTokenizer

# Path to the fine-tuned model
model_path = "path/to/fine_tuning/results/final_model"

# Load the base model and tokenizer
base_model_name = "meta-llama/Llama-3.2-3B-Instruct"
model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    device_map="auto",
    torch_dtype=torch.bfloat16
)
tokenizer = AutoTokenizer.from_pretrained(model_path)

# Load the PEFT adapter
model = PeftModel.from_pretrained(model, model_path)

# Optional: Merge the adapter weights with the base model for faster inference
model = model.merge_and_unload()
```

## Performance Metrics

The fine-tuned model achieves:
- F1 score of 0.74 for dataset identification (compared to 0.65 for the base model)
- Significantly higher accuracy on challenging fields like creator identification (F1 score of 0.59 vs 0.17 for base model)
- Processing time of approximately 35 seconds per paper

## Limitations

The model is specifically fine-tuned for extracting dataset metadata from scientific papers according to the DCAT vocabulary. It may not perform well on other tasks without additional fine-tuning.

## Recovery and Continued Training

If you need to resume training from a checkpoint:

1. Use the `recover_model.py` script in the parent directory
2. Or manually resume training with DeepSpeed, pointing to the checkpoint directory