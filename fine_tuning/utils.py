"""Utility functions for dataset extraction fine-tuning."""

import json
import re
import os
from typing import Dict, List, Union, Optional, Tuple
import numpy as np
from datasets import Dataset

# Model template functions
def get_prompt_template(model_name: str) -> callable:
    """
    Get the appropriate prompt template function based on model name.
    
    Args:
        model_name: Name or path of the model
        
    Returns:
        callable: Function to format prompts for the specified model
    """
    # Map model families to their template functions
    if "llama-3" in model_name.lower():
        return format_llama3_instruct_prompt
    elif "llama-2" in model_name.lower():
        return format_llama2_instruct_prompt
    elif "mistral" in model_name.lower():
        return format_mistral_instruct_prompt
    else:
        # Default to Llama-3 format if unknown
        print(f"Warning: Unknown model {model_name}, defaulting to Llama-3 template")
        return format_llama3_instruct_prompt

def format_llama3_instruct_prompt(
    paper_text: str, 
    system_prompt: Optional[str] = None
) -> str:
    """
    Format prompt according to Llama-3 Instruct template.
    
    Args:
        paper_text: The research paper text
        system_prompt: Optional system prompt
        
    Returns:
        str: Formatted prompt
    """
    if system_prompt is None:
        system_prompt = (
            "You are an expert research assistant specialized in extracting dataset "
            "information from scientific papers. Your task is to identify datasets "
            "mentioned in research papers and describe them using a standardized "
            "metadata vocabulary.\n\n"
            "Follow these steps:\n"
            "1. Carefully analyze the paper to identify all datasets used or mentioned\n"
            "2. For each dataset, extract the required metadata fields\n"
            "3. First provide your reasoning process in a <think> section\n"
            "4. Then output only the formatted JSON result"
        )
    
    user_content = (
        "Extract all datasets mentioned in the research paper below." +
        "Describe each dataset using the following metadata vocabulary and provide the response in JSON format:\n\n"
            "**dcterms:creator**: The entity responsible for creating the dataset. [Look for the list of authors from the dataset reference/citation. For reference style in numbers, use the reference number to locate the dataset (e.g., search \"[32]\" if 32 is the reference number for the dataset). If the style for references looks like this (corpus (Volske et al., 2017)), then search for the dataset in references using the author name \"Volske\".] (if not found, assign empty list, do NOT mix between the papers authors and the dataset authors unless this is really the case. A common mistake is to assign the creators of the datasets as the authors of the article even though the dataset is only used/mentioned in the article. Do NOT consider organizations as creators, creatores should be persons)\n"
            "**dcterms:description**: A textual description of the dataset. [based on how the authors of this paper used or described the dataset]\n"
            "**dcterms:title**: The name of the dataset.\n"
            "**dcterms:issued**: The date/year the dataset was published. [Look for the year in the dataset reference/citation] (if not found, assign empty string)\n"
            "**dcterms:language**: The language of the data in the dataset if the dataset content is a text. [based on what the authors described in this paper, otherwise assign empty string]\n"
            "**dcterms:identifier**: A unique identifier for the dataset (e.g., DOI or URL). [look for dataset doi or url in the dataset reference] (if not found, assign empty string)\n"
            "**dcat:theme**: The thematic category of the dataset. [based on the authors description in this paper. Use a list of categories if more than one category can be assigned, in case of one category use a list with only one category inside it. In case of no categories to be found assign empty list]\n"
            "**dcat:keyword**: Keywords or tags describing the dataset. [generate list of keywords based on the usage and description of the dataset in this paper]\n"
            "**dcat:landingPage**: A web page providing access to the dataset. [look for dataset doi or url in the dataset reference or if it's mentioned in this paper. Otherwise, assign empty string]\n"
            "**dcterms:hasVersion**: A specific version of the dataset. [If mentioned explicitly by the authors, or if there is a number in the name of the dataset that represents the version. Otherwise assign an empty string]\n"
            "**dcterms:format**: The format of the dataset (like Text, Image, Audio,...). [based on how the authors used or described the dataset. Otherwise, assign empty string]\n"
            "**mls:task**: the tasks for which the dataset can be used. [based on the tasks that this paper is tackling. Use a list of tasks for one or more tasks, or assign empty list]\n" +
        "Example of how you should think:\n" +
        """<think>
        To extract datasets from the research paper titled "Sub-word information in pre-trained biomedical word representations: evaluation and hyper-parameter optimization" by Dieter Galea, Ivan Laponogov, and Kirill Veselkov, I will follow a systematic approach.

First, I will read through the **abstract, introduction, materials and methods, and results sections** to identify any datasets that are mentioned or referenced. The abstract mentions the optimization and comparison of representations for the biomedical domain, which suggests that datasets are likely involved.

In the **introduction**, the authors discuss the limitations of word2vec and the importance of character-based representations like fastText. They hint at the evaluation of these models on various datasets, but I need to look for specific names.

Next, I will focus on the **materials and methods section**, particularly the subsections that detail the datasets used for evaluation. Here, the authors mention three named entity recognition corpora:

1. **BioCreative II Gene Mention task corpus (BC2GM)**: This dataset is used for gene recognition tasks.
2. **JNLPBA corpus**: This corpus annotates proteins, cell lines, cell types, DNA, and RNA.
3. **CHEMDNER corpus**: This dataset is focused on annotating drugs and chemicals.

In the **results section**, the authors confirm that these datasets were used for extrinsic evaluation of their models, which reinforces their relevance.

Now, I will check the **references section** to find the full citations for these datasets:

- For **BC2GM**, the citation is:
  > Larry Smith, Lorraine K Tanabe, Rie Johnson nee Ando, Cheng-Ju Kuo, I-Fang Chung, Chun-Nan Hsu, Yu-Shi Lin, Roman Klinger, Christoph M Friedrich, Kuzman Ganchev, et al. *Overview of biocreative ii gene mention recognition*. Genome biology, 9(Suppl 2):1–19, 2008. https://doi.org/10.1186/gb-2008-9-s2-s2

- For **JNLPBA**, the citation is:
  > Jin-Dong Kim, Tomoko Ohta, Yoshimasa Tsuruoka, Yuka Tateisi, and Nigel Collier. *Introduction to the bio-entity recognition task at JNLPBA*. In Proceedings of the 2004 Conference on Computational Natural Language Learning (CoNLL), pages 70–75, 2004. http://www.aclweb.org/anthology/W04-1213

- For **CHEMDNER**, the citation is:
  > Martin Krallinger, Florian Leitner, Obdulia Rabal, Miguel Vazquez, Julen Oyarzabal, and Alfonso Valencia. *CHEMDNER: The drugs and chemical names extraction challenge*. Journal of Cheminformatics, 7(Suppl 1):S1, 2015. https://doi.org/10.1186/1758-2946-7-S1-S1

After gathering this information, I will compile the dataset entries into a structured format for further processing or review. This ensures that I have accurately captured the datasets and their citations from the paper.
</think>

            Example of the dataset description in JSON format:
          [
    {
        "dcterms:creator": [
            "W. Zhang",
            "M. Zhu",
            "K. G. Derpanis"
        ],
        "dcterms:description": "A large dataset containing video clips with 13 joints annotated in all frames, including head, shoulders, elbows, wrists, hips, knees, and ankles.",
        "dcterms:title": "Penn Action Dataset",
        "dcterms:issued": "2013",
        "dcterms:language": "",
        "dcterms:identifier": "",
        "dcat:theme": [
            "Computer Vision",
            "Human Pose Estimation"
        ],
        "dcat:keyword": [
            "Video dataset",
            "Human pose",
            "Joint estimation"
        ],
        "dcat:landingPage": "",
        "dcterms:hasVersion": "",
        "dcterms:format": "Video",
        "mls:task": [
            "Human Pose Estimation"
        ]
    },
""" +
   "Return ONLY the think tag and the json response without any additional messages or text.\n" +     
   "Return only datasets from the input paper. don't try to extract papers or other things. We just need datasets.\n" +     
    "Paper text:\n"
    )
    
    formatted_prompt = (
        "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n"
        f"{system_prompt}\n"
        "<|start_header_id|>user<|end_header_id|>\n"
        f"{user_content}{paper_text}\n"
        "<|start_header_id|>assistant<|end_header_id|>\n"
    )
    
    return formatted_prompt

def format_llama2_instruct_prompt(
    paper_text: str, 
    system_prompt: Optional[str] = None
) -> str:
    """
    Format prompt according to Llama-2 Instruct template.
    
    Args:
        paper_text: The research paper text
        system_prompt: Optional system prompt
        
    Returns:
        str: Formatted prompt
    """
    if system_prompt is None:
        system_prompt = (
            "You are an expert research assistant specialized in extracting dataset "
            "information from scientific papers. Your task is to identify datasets "
            "mentioned in research papers and describe them using a standardized "
            "metadata vocabulary.\n\n"
            "Follow these steps:\n"
            "1. Carefully analyze the paper to identify all datasets used or mentioned\n"
            "2. For each dataset, extract the required metadata fields\n"
            "3. First provide your reasoning process in a <think> section\n"
            "4. Then output only the formatted JSON result"
        )

    instruction = (
        "Extract all datasets mentioned in the research paper below." +
        "Describe each dataset using the following metadata vocabulary and provide the response in JSON format:\n\n"
            "**dcterms:creator**: The entity responsible for creating the dataset. [Look for the list of authors from the dataset reference/citation. For reference style in numbers, use the reference number to locate the dataset (e.g., search \"[32]\" if 32 is the reference number for the dataset). If the style for references looks like this (corpus (Volske et al., 2017)), then search for the dataset in references using the author name \"Volske\".] (if not found, assign empty list, do NOT mix between the papers authors and the dataset authors unless this is really the case. A common mistake is to assign the creators of the datasets as the authors of the article even though the dataset is only used/mentioned in the article. Do NOT consider organizations as creators, creatores should be persons)\n"
            "**dcterms:description**: A textual description of the dataset. [based on how the authors of this paper used or described the dataset]\n"
            "**dcterms:title**: The name of the dataset.\n"
            "**dcterms:issued**: The date/year the dataset was published. [Look for the year in the dataset reference/citation] (if not found, assign empty string)\n"
            "**dcterms:language**: The language of the data in the dataset if the dataset content is a text. [based on what the authors described in this paper, otherwise assign empty string]\n"
            "**dcterms:identifier**: A unique identifier for the dataset (e.g., DOI or URL). [look for dataset doi or url in the dataset reference] (if not found, assign empty string)\n"
            "**dcat:theme**: The thematic category of the dataset. [based on the authors description in this paper. Use a list of categories if more than one category can be assigned, in case of one category use a list with only one category inside it. In case of no categories to be found assign empty list]\n"
            "**dcat:keyword**: Keywords or tags describing the dataset. [generate list of keywords based on the usage and description of the dataset in this paper]\n"
            "**dcat:landingPage**: A web page providing access to the dataset. [look for dataset doi or url in the dataset reference or if it's mentioned in this paper. Otherwise, assign empty string]\n"
            "**dcterms:hasVersion**: A specific version of the dataset. [If mentioned explicitly by the authors, or if there is a number in the name of the dataset that represents the version. Otherwise assign an empty string]\n"
            "**dcterms:format**: The format of the dataset (like Text, Image, Audio,...). [based on how the authors used or described the dataset. Otherwise, assign empty string]\n"
            "**mls:task**: the tasks for which the dataset can be used. [based on the tasks that this paper is tackling. Use a list of tasks for one or more tasks, or assign empty list]\n" +
        "Example of how you should think:\n" +
        """<think>
        To extract datasets from the research paper titled "Sub-word information in pre-trained biomedical word representations: evaluation and hyper-parameter optimization" by Dieter Galea, Ivan Laponogov, and Kirill Veselkov, I will follow a systematic approach.

First, I will read through the **abstract, introduction, materials and methods, and results sections** to identify any datasets that are mentioned or referenced. The abstract mentions the optimization and comparison of representations for the biomedical domain, which suggests that datasets are likely involved.

In the **introduction**, the authors discuss the limitations of word2vec and the importance of character-based representations like fastText. They hint at the evaluation of these models on various datasets, but I need to look for specific names.

Next, I will focus on the **materials and methods section**, particularly the subsections that detail the datasets used for evaluation. Here, the authors mention three named entity recognition corpora:

1. **BioCreative II Gene Mention task corpus (BC2GM)**: This dataset is used for gene recognition tasks.
2. **JNLPBA corpus**: This corpus annotates proteins, cell lines, cell types, DNA, and RNA.
3. **CHEMDNER corpus**: This dataset is focused on annotating drugs and chemicals.

In the **results section**, the authors confirm that these datasets were used for extrinsic evaluation of their models, which reinforces their relevance.

Now, I will check the **references section** to find the full citations for these datasets:

- For **BC2GM**, the citation is:
  > Larry Smith, Lorraine K Tanabe, Rie Johnson nee Ando, Cheng-Ju Kuo, I-Fang Chung, Chun-Nan Hsu, Yu-Shi Lin, Roman Klinger, Christoph M Friedrich, Kuzman Ganchev, et al. *Overview of biocreative ii gene mention recognition*. Genome biology, 9(Suppl 2):1–19, 2008. https://doi.org/10.1186/gb-2008-9-s2-s2

- For **JNLPBA**, the citation is:
  > Jin-Dong Kim, Tomoko Ohta, Yoshimasa Tsuruoka, Yuka Tateisi, and Nigel Collier. *Introduction to the bio-entity recognition task at JNLPBA*. In Proceedings of the 2004 Conference on Computational Natural Language Learning (CoNLL), pages 70–75, 2004. http://www.aclweb.org/anthology/W04-1213

- For **CHEMDNER**, the citation is:
  > Martin Krallinger, Florian Leitner, Obdulia Rabal, Miguel Vazquez, Julen Oyarzabal, and Alfonso Valencia. *CHEMDNER: The drugs and chemical names extraction challenge*. Journal of Cheminformatics, 7(Suppl 1):S1, 2015. https://doi.org/10.1186/1758-2946-7-S1-S1

After gathering this information, I will compile the dataset entries into a structured format for further processing or review. This ensures that I have accurately captured the datasets and their citations from the paper.
</think>

            Example of the dataset description in JSON format:
          [
    {
        "dcterms:creator": [
            "W. Zhang",
            "M. Zhu",
            "K. G. Derpanis"
        ],
        "dcterms:description": "A large dataset containing video clips with 13 joints annotated in all frames, including head, shoulders, elbows, wrists, hips, knees, and ankles.",
        "dcterms:title": "Penn Action Dataset",
        "dcterms:issued": "2013",
        "dcterms:language": "",
        "dcterms:identifier": "",
        "dcat:theme": [
            "Computer Vision",
            "Human Pose Estimation"
        ],
        "dcat:keyword": [
            "Video dataset",
            "Human pose",
            "Joint estimation"
        ],
        "dcat:landingPage": "",
        "dcterms:hasVersion": "",
        "dcterms:format": "Video",
        "mls:task": [
            "Human Pose Estimation"
        ]
    },
""" +
   "Return ONLY the think tag and the json response without any additional messages or text.\n" +     
   "Return only datasets from the input paper. don't try to extract papers or other things. We just need datasets.\n" +     
    "Paper text:\n"
    )

    
    formatted_prompt = (
        f"<s>[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n\n"
        f"{instruction}\n\nPaper text:\n{paper_text} [/INST]"
    )
    
    return formatted_prompt

def format_mistral_instruct_prompt(
    paper_text: str, 
    system_prompt: Optional[str] = None
) -> str:
    """
    Format prompt according to Mistral Instruct template.
    
    Args:
        paper_text: The research paper text
        system_prompt: Optional system prompt
        
    Returns:
        str: Formatted prompt
    """
    if system_prompt is None:
        system_prompt = (
            "You are an expert research assistant specialized in extracting dataset "
            "information from scientific papers. Your task is to identify datasets "
            "mentioned in research papers and describe them using a standardized "
            "metadata vocabulary."
        )
    
    formatted_prompt = (
        f"<s>[INST] {system_prompt}\n\n"
        "Extract all datasets mentioned in this research paper. Describe each dataset "
        "using the following metadata vocabulary and provide the response in JSON format:\n\n"
        "**dcterms:creator**: The entity responsible for creating the dataset. "
        "[List of authors from the dataset citation; not the paper authors unless they created the dataset]\n\n"
        "**dcterms:description**: A textual description of the dataset based on how it's described in the paper.\n\n"
        "**dcterms:title**: The name of the dataset.\n\n"
        "**dcterms:issued**: The year the dataset was published. [From citation]\n\n"
        "**dcterms:language**: The language of the dataset content, if text-based.\n\n"
        "**dcterms:identifier**: A unique identifier like DOI or URL.\n\n"
        "**dcat:theme**: The thematic category/categories of the dataset.\n\n"
        "**dcat:keyword**: Keywords describing the dataset based on its usage in the paper.\n\n"
        "**dcat:landingPage**: Web page providing access to the dataset.\n\n"
        "**dcterms:hasVersion**: The specific version of the dataset, if mentioned.\n\n"
        "**dcterms:format**: The format of the dataset (Text, Image, Audio, etc.).\n\n"
        "**mls:task**: Tasks for which the dataset can be used, based on the paper.\n\n"
        f"Paper text:\n{paper_text} [/INST]"
    )
    
    return formatted_prompt

# Evaluation functions
def extract_json_from_response(response: str) -> Optional[str]:
    """
    Extract JSON content from model response.
    
    Args:
        response: Model response text
        
    Returns:
        Optional[str]: Extracted JSON string or None if not found
    """
    # Try to extract JSON from markdown code blocks
    json_match = re.search(r'```(?:json)?\s*(\[[\s\S]*?\])\s*```', response)
    if json_match:
        return json_match.group(1)
    
    # Try to extract raw JSON array
    json_match = re.search(r'(\[\s*\{[\s\S]*?\}\s*\])', response)
    if json_match:
        return json_match.group(1)
    
    return None

def compute_json_accuracy(predictions: List[str], references: List[str]) -> Dict[str, float]:
    """
    Compute metrics for JSON prediction accuracy.
    
    Args:
        predictions: List of model prediction texts
        references: List of reference JSON strings
        
    Returns:
        Dict[str, float]: Dictionary of evaluation metrics
    """
    results = {
        "valid_json": 0.0,
        "structure_match": 0.0,
        "field_presence": {},
        "field_exact_match": {},
    }
    
    total = len(predictions)
    valid_count = 0
    structure_match_count = 0
    field_presence = {}
    field_match = {}
    
    for pred, ref in zip(predictions, references):
        # Extract JSON from prediction
        pred_json_str = extract_json_from_response(pred)
        if not pred_json_str:
            continue
            
        try:
            pred_json = json.loads(pred_json_str)
            ref_json = json.loads(ref)
            valid_count += 1
            
            # Check if top-level structure is correct (array of objects)
            if isinstance(pred_json, list) and all(isinstance(item, dict) for item in pred_json):
                structure_match_count += 1
                
                # For each dataset in the prediction
                for dataset in pred_json:
                    # Count field presence
                    for field in [
                        "dcterms:creator", "dcterms:description", "dcterms:title",
                        "dcterms:issued", "dcterms:language", "dcterms:identifier",
                        "dcat:theme", "dcat:keyword", "dcat:landingPage",
                        "dcterms:hasVersion", "dcterms:format", "mls:task"
                    ]:
                        field_presence[field] = field_presence.get(field, 0) + (1 if field in dataset else 0)
            
            # Check for exact matches between predicted and reference datasets
            # This is simplified - a more robust implementation would match datasets by title
            # and check field-by-field similarity
            if len(pred_json) == len(ref_json):
                for i, ref_dataset in enumerate(ref_json):
                    if i < len(pred_json):
                        pred_dataset = pred_json[i]
                        for field, value in ref_dataset.items():
                            field_match[field] = field_match.get(field, 0)
                            if field in pred_dataset and pred_dataset[field] == value:
                                field_match[field] += 1
            
        except json.JSONDecodeError:
            continue
    
    # Calculate metrics
    results["valid_json"] = valid_count / total if total > 0 else 0
    results["structure_match"] = structure_match_count / total if total > 0 else 0
    
    # Calculate field presence percentages
    for field, count in field_presence.items():
        results["field_presence"][field] = count / valid_count if valid_count > 0 else 0
    
    # Calculate field exact match percentages
    for field, count in field_match.items():
        results["field_exact_match"][field] = count / valid_count if valid_count > 0 else 0
    
    return results

def extract_thinking_from_response(response: str) -> Optional[str]:
    """
    Extract thinking content from model response.
    
    Args:
        response: Model response text
        
    Returns:
        Optional[str]: Extracted thinking text or None if not found
    """
    thinking_match = re.search(r'<think>([\s\S]*?)</think>', response)
    if thinking_match:
        return thinking_match.group(1).strip()
    
    return None

def create_length_bins(dataset: Dataset, input_col: str, num_bins: int = 15) -> Dataset:
    """
    Add length bins to dataset for more efficient batching.
    Uses more fine-grained bins for better memory usage during training.
    
    Args:
        dataset: Input dataset
        input_col: Column name containing input text or tokens
        num_bins: Number of length bins to create
        
    Returns:
        Dataset: Dataset with length bin column added
    """
    # Calculate text/token lengths
    lengths = [len(x) for x in dataset[input_col]]
    
    # Create exponential bin edges for better handling of long-tail distribution
    # This gives more fine-grained bins for shorter sequences and wider bins for longer ones
    max_length = max(lengths)
    min_length = min(lengths)
    
    # Create bin edges with exponential scaling
    if max_length > 2 * min_length:
        # Use exponential bins for skewed distributions
        log_min = np.log(min_length) if min_length > 0 else 0
        log_max = np.log(max_length)
        bin_edges = np.exp(np.linspace(log_min, log_max, num_bins + 1))
    else:
        # Use linear bins for more uniform distributions
        bin_edges = np.linspace(min_length, max_length, num_bins + 1)
    
    # Ensure bin edges are integers
    bin_edges = np.unique(bin_edges.astype(int))
    
    # If we ended up with fewer bins after uniqueness, adjust
    if len(bin_edges) < num_bins + 1:
        # Add additional evenly spaced bins
        num_bins = len(bin_edges) - 1
    
    # Assign bin to each example
    bins = []
    for length in lengths:
        for i in range(len(bin_edges) - 1):
            if length >= bin_edges[i] and (i == len(bin_edges) - 2 or length < bin_edges[i + 1]):
                bins.append(i)
                break
        else:
            bins.append(num_bins - 1)  # Fallback
    
    # Add bin column to dataset
    dataset = dataset.add_column("length_bin", bins)
    
    # Print bin statistics
    bin_counts = {}
    for bin_idx in range(num_bins):
        bin_counts[bin_idx] = bins.count(bin_idx)
    
    print(f"Created {num_bins} length bins with distribution: {bin_counts}")
    
    return dataset