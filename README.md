# MetaMine: A Neuro-Symbolic Approach for Dataset Extraction from Research Papers

MetaMine is a novel approach for extracting structured dataset metadata from scientific research papers. It utilizes a multi-stage chain-of-thought prompting strategy with knowledge distillation to train a compact model that can accurately identify and extract dataset metadata according to the DCAT vocabulary standard.

> **Note on large files**: Some files in the `data/aws/` directory have been split into smaller chunks to comply with GitHub's file size limits. To work with these files:
> 1. Use the individual part files directly, or
> 2. Combine them back using the provided script: `python combine_csv_files.py --input_dir data/aws/split --output_dir data/aws/combined`

## Project Overview

Scientific datasets are valuable knowledge assets often hidden within research papers, limiting their discovery and reuse. MetaMine addresses this challenge by:

1. Using a multi-stage chain-of-thought prompting strategy to guide large teacher models (GPT) in dataset identification and metadata extraction
2. Employing knowledge distillation to transfer these capabilities to a smaller student model (Llama-3.2-3B-Instruct)
3. Preserving the reasoning process during distillation for improved extraction accuracy
4. Aligning extracted metadata with the DCAT vocabulary for semantic web integration
5. Converting the structured output into RDF triples for knowledge graph creation

The distilled model processes papers in 35 seconds compared to 120 seconds for larger models, making it practical for processing large scientific corpora while maintaining high-quality extraction.

## Pipeline Structure

The MetaMine pipeline consists of four main phases:

1. **Data Collection and Processing**: Papers are collected from sources like Papers With Code and processed through OCR to extract text content.
2. **Data Annotation**: A teacher model (GPT-o4-mini) annotates papers using a multi-stage prompting strategy, and a subset is verified by human annotators.
3. **Knowledge Distillation**: The extraction capabilities and reasoning process are transferred to a smaller student model (Llama-3.2-3B-Instruct) through fine-tuning.
4. **Knowledge Graph Creation**: Extracted metadata is converted to RDF triples for integration with the semantic web.

## Directory Structure

- **data/**: Contains all data files
  - **aws/**: Amazon Mechanical Turk annotation files
  - **gs/**: Gold standard datasets
  - **llama/**: Generated output from the base Llama model
  - **llama_tuned/**: Generated output from the fine-tuned Llama model
  - **qwen/**: Generated output from the DeepSeek Qwen model
- **fine_tuning/**: Scripts for fine-tuning the student model
- **inference/**: Scripts for generating dataset metadata using the fine-tuned model
- **results/**: Evaluation results for different models

## Installation and Dependencies

The project requires the following dependencies:
- Python 3.8+
- PyTorch
- Transformers
- PEFT (Parameter-Efficient Fine-Tuning)
- DeepSpeed
- Pandas
- Matplotlib
- pdfminer.six

## Usage

The MetaMine pipeline consists of the following main scripts:

1. **1_choose_papers_randomly.py**: Selects papers randomly from Papers With Code repository
2. **2_download_papers.py**: Downloads selected papers in PDF format
3. **3_pdf2txt.py**: Converts PDF files to text
4. **4_process_papers_fine_tune.py**: Processes papers with the teacher model to generate training data
5. **5_combine_datasets.py**: Merges the output of different models into a single dataset file
6. **6_combine_csv.py**: Processes Amazon Mechanical Turk annotations
7. **7_annotation_accuracy.py**: Analyzes annotation accuracy and generates figures
8. **8_order_columns.py**: Reorders columns in annotation files for better readability

### Fine-tuning the Model

To fine-tune the model, navigate to the `fine_tuning` directory and run:

```bash
bash run_fine_tune.sh
```

### Running Inference

To extract dataset metadata from new papers:

```bash
cd inference
bash inference.sh
```

## Results

The distilled model (Llama-3.2-3B-Instruct) achieves an F1 score of 0.74 for dataset identification, outperforming its pre-distillation baseline (0.65) and rivaling much larger models like DeepSeek-R1-Distill-Qwen-32B (0.73) despite being 10× smaller. The model particularly excels at challenging metadata fields like dataset creator identification.

## Citation

If you use MetaMine in your research, please cite:

```
@inproceedings{metamine2023,
  title={MetaMine: A Neuro-Symbolic Approach for Dataset Extraction from Research Papers using Knowledge Distillation},
  author={Anonymous Author(s)},
  booktitle={Anonymous Conference},
  year={2023}
}
```

## License

This project is licensed under [License Name] - see the LICENSE file for details.