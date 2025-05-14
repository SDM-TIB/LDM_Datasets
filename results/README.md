# Results Directory

This directory contains the evaluation results for different models tested in the MetaMine project.

## Files

- **DeepSeek(QWEN Distill).json**: Evaluation results for the DeepSeek-R1-Distill-Qwen-32B model
- **Student(Llama)-base.json**: Evaluation results for the base Llama-3.2-3B-Instruct model without fine-tuning
- **Student(Llama)-tuned-wo-think.json**: Evaluation results for the Llama-3.2-3B-Instruct model fine-tuned without the reasoning step preservation
- **Student(Llama)-tuned.json**: Evaluation results for the full MetaMine model (Llama-3.2-3B-Instruct fine-tuned with reasoning preservation)

## Evaluation Metrics

Each result file contains performance metrics for:

1. **Dataset Identification Coverage**: How accurately the model identifies datasets mentioned in papers
   - Precision, Recall, and F1 score

2. **Field-Specific Performance**: How accurately the model extracts individual metadata fields
   - Performance for fields like dataset creator, theme, keyword, task, title, publication date, etc.

3. **Efficiency Metrics**: How efficiently the model processes papers
   - Processing time per paper

## Key Findings

The evaluation results demonstrate:

1. The fine-tuned MetaMine model (Student(Llama)-tuned) achieves an F1 score of 0.74 for dataset identification, outperforming its pre-distillation baseline (0.65) and rivaling the much larger DeepSeek model (0.73) despite being 10× smaller.

2. The importance of reasoning preservation during knowledge distillation, as shown by the significant performance gap between the model with thinking capabilities (Student(Llama)-tuned) and the one without (Student(Llama)-tuned-wo-think).

3. The MetaMine model particularly excels at challenging metadata fields like dataset creator identification, where it achieves an F1 score of 0.59 compared to DeepSeek's 0.39 and the base model's 0.17.

4. The distilled model processes papers in 35 seconds compared to 120 seconds for the 32B DeepSeek model, making it much more efficient for large-scale processing.