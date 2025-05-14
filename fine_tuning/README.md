# 🧠 Fine-Tuning Directory

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

This directory contains scripts and configuration files for fine-tuning the student model (Llama-3.2-3B-Instruct) using knowledge distillation from the teacher model (GPT-o4-mini).

## 📋 Files

- **ds_config.json**: DeepSpeed configuration file for distributed training
- **preprocess.py**: Script to preprocess the data for fine-tuning
- **recover_model.py**: Script to recover a model from checkpoints
- **recover_model.sh**: Shell script to run the model recovery process
- **run_fine_tune.sh**: Main shell script to run the fine-tuning process
- **train.py**: Python script for the actual training process
- **utils.py**: Utility functions for training and evaluation

## 🔄 Fine-Tuning Process

The fine-tuning process transfers knowledge from the teacher model (GPT-o4-mini) to the student model (Llama-3.2-3B-Instruct) using Parameter-Efficient Fine-Tuning (PEFT) with LoRA. The student model learns to:

1. Identify datasets mentioned in scientific papers
2. Extract structured metadata according to the DCAT vocabulary
3. Preserve the reasoning process through the use of the `<think>` tag

## ⚙️ Configuration

The fine-tuning process uses the following hyperparameters:

- **LoRA Configuration**:
  - Rank: 16
  - Alpha: 32
  - Dropout: 0.05
  - Target modules: q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj

- **Training Configuration**:
  - Number of epochs: 3
  - Learning rate: 2e-4
  - Weight decay: 0.01
  - Warmup ratio: 0.1
  - Gradient accumulation steps: 32
  - Batch size: 1

## 🚀 Running Fine-Tuning

To run the fine-tuning process:

```bash
bash run_fine_tune.sh
```

This script will:
1. Preprocess the data
2. Initialize the model with 4-bit quantization
3. Configure LoRA for parameter-efficient fine-tuning
4. Run the training with DeepSpeed for distributed training
5. Save the fine-tuned model and its checkpoints

## 💻 Hardware Requirements

The fine-tuning process is designed to run on 4 NVIDIA H100 GPUs. The script is configured to run on a SLURM cluster with specific partition requirements.

## 🔄 Recovering a Model

If the fine-tuning process is interrupted, you can recover the model from checkpoints:

```bash
bash recover_model.sh
```

This will use the latest checkpoint to continue the fine-tuning process.