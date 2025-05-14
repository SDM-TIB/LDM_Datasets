# Gold Standard (gs) Directory

This directory contains the gold standard dataset used for training and evaluating the MetaMine models. It represents a curated collection of papers and their associated dataset metadata, verified by human annotators.

## Directory Structure

- **combined_papers_metadata.csv**: Comprehensive metadata for all papers in the gold standard
- **datasets.json**: Combined dataset information from all sources, merged using the rules defined in `5_combine_datasets.py`
- **downloaded_papers.json**: Information about all downloaded papers, including their unique IDs and URLs
- **json_files/**: Directory containing extracted dataset information
  - **Computer_Vision/**: Extracted datasets from computer vision papers
  - **General/**: Extracted datasets from general ML papers
  - **Graphs/**: Extracted datasets from graph-related papers
  - **Natural_Language_Processing/**: Extracted datasets from NLP papers
- **paper_urls.txt**: URLs of papers used in the gold standard
- **pdfs/**: Directory for downloaded PDF files (not included due to copyright restrictions)
- **processing_summary.json**: Summary of the processing statistics
- **txt_files/**: Directory for extracted text from PDFs (not included due to copyright restrictions)

## JSON File Format

Each JSON file in the `json_files` subdirectories contains structured dataset information following the DCAT vocabulary. For each paper, there are two files:

1. **paper_X.json**: Contains structured metadata about datasets mentioned in the paper, including:
   - `dcterms:creator`: List of authors who created the dataset
   - `dcterms:description`: Textual description of the dataset
   - `dcterms:title`: Name of the dataset
   - `dcterms:issued`: Publication year of the dataset
   - `dcterms:language`: Language of the dataset (if applicable)
   - `dcterms:identifier`: DOI or URL identifier for the dataset
   - `dcat:theme`: Thematic categories of the dataset
   - `dcat:keyword`: Keywords associated with the dataset
   - `dcat:landingPage`: Web page for accessing the dataset
   - `dcterms:hasVersion`: Version information
   - `dcterms:format`: Format of the dataset (e.g., Text, Image, Audio)
   - `mls:task`: Tasks for which the dataset is used

2. **paper_X_thinking.txt**: Contains the reasoning process that the teacher model used to extract the dataset information

## PDF and TXT Files

The `pdfs/` and `txt_files/` directories are placeholders for the downloaded PDF files and their extracted text. These files are not included in the repository due to copyright restrictions. To replicate the project, you would need to:

1. Use `1_choose_papers_randomly.py` to select papers from Papers With Code
2. Use `2_download_papers.py` to download the selected papers
3. Use `3_pdf2txt.py` to convert the PDFs to text

## Creation Process

The gold standard was created through the following steps:

1. Random selection of papers from different areas (NLP, CV, Graphs, etc.)
2. Downloading and converting papers to text
3. Processing papers with the teacher model (GPT-o4-mini)
4. Manual verification by Amazon Mechanical Turk workers
5. Resolving conflicts through majority voting
6. Combining and standardizing the dataset information

This gold standard serves as the benchmark for evaluating the performance of different models in extracting dataset metadata from scientific papers.