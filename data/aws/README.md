# AWS Directory - Amazon Mechanical Turk Annotations

This directory contains annotation files from Amazon Mechanical Turk (AMT) workers who verified the dataset metadata extracted by the teacher model.

## Files

- **Batch_5281193_batch_results.csv**: Raw annotation results from the first batch of HITs (split into parts for GitHub)
  - **split/Batch_5281193_batch_results/**: Contains the file split into smaller chunks under 100MB:
    - **Batch_5281193_batch_results_part1.csv**: First 1021 rows (68.73 MB)
    - **Batch_5281193_batch_results_part2.csv**: Remaining 1020 rows (60.84 MB)
- **Batch_5281193_reordered.csv**: Same data as above but with columns reordered for better readability (split into parts for GitHub)
  - **split/Batch_5281193_reordered/**: Contains the file split into smaller chunks under 100MB:
    - **Batch_5281193_reordered_part1.csv**: First 1021 rows (68.73 MB)
    - **Batch_5281193_reordered_part2.csv**: Remaining 1020 rows (60.84 MB)
- **Batch_5282286_batch_results.csv**: Raw annotation results from the second batch of HITs (split into parts for GitHub)
  - **split/Batch_5282286_batch_results/**: Contains the file split into smaller chunks under 100MB:
    - **Batch_5282286_batch_results_part1.csv**: First 1473 rows (81.69 MB)
    - **Batch_5282286_batch_results_part2.csv**: Remaining 1473 rows (76.22 MB)
- **Batch_5282286_reordered.csv**: Same data as above but with columns reordered for better readability (split into parts for GitHub)
  - **split/Batch_5282286_reordered/**: Contains the file split into smaller chunks under 100MB:
    - **Batch_5282286_reordered_part1.csv**: First 1473 rows (81.69 MB)
    - **Batch_5282286_reordered_part2.csv**: Remaining 1473 rows (76.22 MB)

## Annotation Process

For the gold standard creation, Amazon Mechanical Turk workers were tasked with verifying the dataset metadata extracted by the teacher model (GPT-o4-mini). The verification process involved:

1. Assigning three independent workers to review each extracted metadata field
2. Instructing workers to evaluate the accuracy of each field and correct any errors
3. Resolving conflicts through majority voting when workers disagreed on corrections
4. Ensuring alignment with the DCAT vocabulary across all verified entries

## Data Structure

The annotation files contain the following types of columns:

- **HITId**: Unique identifier for the Human Intelligence Task
- **WorkerId**: Anonymous identifier for the AMT worker
- **AssignmentStatus**: Whether the assignment was Approved or Rejected
- **WorkTimeInSeconds**: Time taken by the worker to complete the task
- **LifetimeApprovalRate**: Worker's overall approval rate on AMT
- **Input.paper**: Reference to the paper being annotated
- **Input.dcterms_X**: Original metadata extracted by the teacher model
- **Answer.corrected_X**: Corrected metadata provided by the annotator
- **Answer.X_correct.True/False**: Boolean indicating whether the original metadata was correct

Where X represents different metadata fields:
- accessRights
- creator
- date (issued)
- description
- format
- identifier
- keywords
- landingPage
- language
- task
- theme
- title
- type
- version (hasVersion)

## Usage

These annotations serve multiple purposes:

1. Creating a gold standard for evaluating the performance of different models
2. Providing verified training data for fine-tuning the student model
3. Analyzing the accuracy of the teacher model's extraction capabilities

The files can be processed using the `7_annotation_accuracy.py` script to compute annotation accuracy for each field, and the `8_order_columns.py` script to reorder columns for better readability.