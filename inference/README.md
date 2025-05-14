# 🔍 Inference Directory

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

This directory contains scripts for generating dataset metadata from scientific papers using the fine-tuned models.

## 📋 Files

- **environment.yaml**: Conda environment specification for inference
- **inference.sh**: Shell script to run the inference process
- **process_papers.py**: Main Python script for processing papers with different models
- **process_papers_ds.py**: Script for processing papers with DeepSeek models
- **process_papers_ft.py**: Script for processing papers with fine-tuned models

## 🤖 Models

The inference scripts support multiple models:

1. **Llama-3.2-3B-Instruct (base)**: The base model without fine-tuning
2. **Llama-3.2-3B-Instruct (tuned)**: The model fine-tuned with knowledge distillation
3. **DeepSeek-R1-Distill-Qwen-32B**: A larger (32B parameter) model for comparison

## 🔄 Inference Process

The inference process follows these steps:

1. Load the specified model with appropriate configurations
2. Preprocess the input papers to fit within the model's context window
3. Generate dataset metadata using a chain-of-thought approach
4. Extract structured JSON output and reasoning process
5. Save the results to output files

## 🚀 Usage

To run inference with a specific model:

```bash
bash inference.sh --model MODEL_NAME --input_folder INPUT_DIR --output_folder OUTPUT_DIR
```

Where:
- `MODEL_NAME`: The name or path of the model to use (e.g., "meta-llama/Llama-3.2-3B-Instruct")
- `INPUT_DIR`: Directory containing text files of papers to process
- `OUTPUT_DIR`: Directory to save the output JSON files

## 📊 Output Format

For each input paper, the inference process generates two files:

1. **paper_X.json**: JSON file containing structured dataset metadata according to the DCAT vocabulary
2. **paper_X_thinking.txt**: Text file containing the reasoning process that led to the metadata extraction

## ⚡ Performance Considerations

The scripts are optimized for efficient inference:
- Models are loaded with 4-bit quantization to reduce memory requirements
- Flash Attention 2 is used when available for faster processing
- Torch compilation is applied for better performance
- The scripts handle CUDA out-of-memory errors gracefully by reinitializing the model