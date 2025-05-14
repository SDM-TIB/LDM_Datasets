import os
import json
import logging
from datetime import datetime
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch
from tqdm import tqdm
from huggingface_hub import login
import multiprocessing as mp
import re
from peft import PeftModel

# Replace with your Hugging Face API token
login("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")

# ============================
# 1. Setup and Configuration
# ============================
model_name_='llama_tuned'
logging.basicConfig(
    filename= "logs/"+model_name_+"_generation.log",
    filemode='w',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

logging.info("Starting generation process.")

# Input and output directories
#input_folder = 'data/txt_files/Test2'
#output_folder = 'data/json_files/Test2'

errors_folder = 'data/'+model_name_+'/errors_files'
error_log = 'logs/errors_processing_'+model_name_+'.log'
#os.makedirs(output_folder, exist_ok=True)
os.makedirs(errors_folder, exist_ok=True)
os.makedirs(os.path.dirname(error_log), exist_ok=True)

# System and template prompts
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

# ============================
# 2. Model Family Configuration
# ============================

# Define a class to manage model families and their prompt formats
class ModelFamily:
    # Known model families and their identifiers
    FAMILIES = {
        'llama': ['meta-llama', 'llama', 'mixtral', 'mistral'],
        'deepseek': ['deepseek'],
    }
    
    # Prompt templates for different model families
    TEMPLATES = {
        'llama': {
            'system_prefix': '<|start_header_id|>system<|end_header_id|>',
            'system_suffix': '<|eot_id|>',
            'user_prefix': '<|start_header_id|>user<|end_header_id|>',
            'user_suffix': '<|eot_id|>',
            'assistant_prefix': '<|start_header_id|>assistant<|end_header_id|>',
            'assistant_suffix': ''
        },
        'deepseek': {
            'system_prefix': '',
            'system_suffix': '',
            'user_prefix': '<｜User｜>',
            'user_suffix': '',
            'assistant_prefix': '<｜Assistant｜>',
            'assistant_suffix': '<｜end▁of▁sentence｜>'
        }
        ,
        'default': {
            'system_prefix': '<|start_header_id|>system<|end_header_id|>',
            'system_suffix': '<|eot_id|>',
            'user_prefix': '<|start_header_id|>user<|end_header_id|>',
            'user_suffix': '<|eot_id|>',
            'assistant_prefix': '<|start_header_id|>assistant<|end_header_id|>',
            'assistant_suffix': ''
        }
    }
    
    @classmethod
    def detect_family(cls, model_id):
        """Detect the model family from the model ID."""
        model_id_lower = model_id.lower()
        
        for family, identifiers in cls.FAMILIES.items():
            for identifier in identifiers:
                if identifier in model_id_lower:
                    #logging.info(f"model family: {family}")
                    return family
        
        logging.warning(f"Unknown model family for {model_id}, using default template")
        return 'default'
    
    @classmethod
    def get_template(cls, model_id):
        """Get the template for the given model ID."""
        family = cls.detect_family(model_id)
        return cls.TEMPLATES.get(family, cls.TEMPLATES['default'])
    
    @classmethod
    def format_prompt(cls, model_id, system_prompt, user_prompt, paper_text):
        """Format the prompt according to the model's template."""
        template = cls.get_template(model_id)
        #logging.info(f"template user_prefix: {template['user_prefix']}")
        
        
        full_user_prompt = f"{paper_text}\n\n{user_prompt}"
        
        formatted_prompt = (
            f"{template['system_prefix']}{system_prompt}{template['system_suffix']}"
            f"{template['user_prefix']}{full_user_prompt}{template['user_suffix']}"
            f"{template['assistant_prefix']}"
        )
        
        return formatted_prompt, template['assistant_suffix']

# ============================
# 3. Initialize the Model Pipeline
# ============================

def initialize_model(model_id):
    try:
        dtype = torch.bfloat16  # BF16 is generally good for H100s
        fine_tuned_model_id = "../fine_tune/results/final_model"
        
        # Load the tokenizer with the special tokens
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = 'left'  # Set padding side to 'left' for decoder-only models
        
        quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
        
        # Load the model with device_map="auto" to automatically distribute across available GPUs
        base_model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=dtype,
            device_map="auto",  # Automatically distributes across available GPUs
            attn_implementation="flash_attention_2"  # Use FlashAttention2 for better performance
            #quantization_config=quantization_config
        )
        model = PeftModel.from_pretrained(
                base_model,
                fine_tuned_model_id,
                torch_dtype=dtype,
                device_map="auto"
            )

        model = torch.compile(model) # Compile the model for better performance
        
        # Create the pipeline with the updated model and tokenizer
        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            torch_dtype=dtype,
        )
        
        logging.info(f"Model '{model_id}' loaded successfully across multiple GPUs.")
        return pipe
    except Exception as e:
        logging.error(f"Failed to load model '{model_id}': {e}")
        raise e

# ============================
# 4. Prepare Prompts from Input Files
# ============================

def clean_text(text: str) -> str:
    """
    Clean text by removing unwanted unicode artifacts, (cid:XX) patterns,
    and explicit Unicode escape sequences.
    """
    import unicodedata
    
    # Remove (cid:XX) artifacts
    text = re.sub(r'\(cid:\d+\)', '', text)
    
    # First attempt to decode Unicode escape sequences
    try:
        text = text.encode('utf-8').decode('unicode_escape')
    except Exception:
        pass  # If decoding fails, ignore
    
    # Remove any literal Unicode escape sequences that remain as text
    text = re.sub(r'\\u[0-9a-fA-F]{4}', '', text)
    
    # Normalize Unicode characters
    text = unicodedata.normalize('NFKD', text)
    
    # Use ASCII-only encoding to remove all non-ASCII characters
    text = text.encode('ascii', 'ignore').decode('ascii')
    
    # Remove any remaining non-printable characters
    text = ''.join(c for c in text if c.isprintable())
    
    # Collapse multiple spaces
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text
    
    
def collect_file_paths(input_folder):
    file_paths = []
    for root, dirs, files in os.walk(input_folder):
        for filename in files:
            if filename.endswith('.txt'):
                relative_path = os.path.relpath(os.path.join(root, filename), input_folder)
                file_paths.append(relative_path)
    return file_paths

def prepare_prompts(file_paths, input_folder, output_folder, errors_folder, system_prompt, user_content, model_id, max_tokens=128000):
    data = []
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    
    for relative_path in file_paths:
        input_path = os.path.join(input_folder, relative_path)
        output_path = os.path.join(output_folder, relative_path.replace('.txt', '.json'))
        output_dir = os.path.dirname(output_path)
        os.makedirs(output_dir, exist_ok=True)

        file_id = os.path.splitext(os.path.basename(relative_path))[0]
        error_file_path = os.path.join(errors_folder, f"{file_id}.txt")

        # Skip if already processed
        if os.path.exists(output_path) or os.path.exists(error_file_path):
            continue

        try:
            # Read the paper text
            with open(input_path, 'r', encoding='utf-8') as file:
                paper_text = file.read().strip('"').strip("'")
            
            paper_text = clean_text(paper_text)
            # Format the prompt according to the model's template
            full_prompt, assistant_suffix = ModelFamily.format_prompt(
                model_id, system_prompt, user_content, paper_text
            )
            
            # Check token length
            tokens = tokenizer.encode(full_prompt)
            
            # If tokens exceed the limit, truncate paper_text from the beginning
            if len(tokens) > max_tokens:
                logging.warning(f"File {relative_path} exceeds token limit: {len(tokens)} > {max_tokens}")
                
                # Calculate how much to truncate
                # Create a version of the prompt without the paper text to determine its token count
                template_no_paper, _ = ModelFamily.format_prompt(
                    model_id, system_prompt, user_content, ""
                )
                template_tokens = len(tokenizer.encode(template_no_paper))
                
                # Available tokens for paper_text
                available_tokens = max_tokens - template_tokens
                
                # Tokenize paper_text and truncate from beginning
                paper_tokens = tokenizer.encode(paper_text)
                if len(paper_tokens) > available_tokens:
                    truncated_paper_tokens = paper_tokens[-available_tokens:]
                    truncated_paper = tokenizer.decode(truncated_paper_tokens)
                    
                    # Reconstruct the prompt with truncated paper
                    full_prompt, assistant_suffix = ModelFamily.format_prompt(
                        model_id, system_prompt, user_content, truncated_paper
                    )
                    
                    # Log truncation info
                    percent_kept = (len(truncated_paper_tokens) / len(paper_tokens)) * 100
                    logging.info(f"Truncated {relative_path}: Kept {percent_kept:.1f}% of paper text, final token count: {len(tokenizer.encode(full_prompt))}")
            
            data.append({
                'prompt': full_prompt,
                'assistant_suffix': assistant_suffix,
                'output_path': output_path,
                'error_file_path': error_file_path,
                'relative_path': relative_path,
                'file_id': file_id,
            })

        except Exception as e:
            logging.error(f"Error preparing prompt for {relative_path}: {e}")
            continue

    return data

# ============================
# 5. Generate Outputs and Save
# ============================

def process_model_response(generated_text, output_path, thinking_path):
    """Process the model's response to extract JSON and thinking parts."""
    # Split thinking and JSON response
    if "</think>" in generated_text:
        thinking_part, json_part = generated_text.split("</think>", 1)
        thinking_part += "</think>"  # Add back the tag for completeness
    else:
        # Fallback if the thinking part isn't structured as expected
        thinking_part = ""
        json_part = generated_text
    
    # Clean up the JSON part to extract valid JSON
    json_part = json_part.strip()
    
    # Extract JSON content from the text
    # Look for a JSON array pattern
    json_matches = re.findall(r'\[\s*\{.*\}\s*\]', json_part, re.DOTALL)
    if json_matches:
        json_content = json_matches[0]
    else:
        # Fallback to look for JSON objects without array
        json_matches = re.findall(r'\{.*\}', json_part, re.DOTALL)
        if json_matches:
            json_content = json_matches[0]
        else:
            raise ValueError("No valid JSON found in the response")
    
    # Parse and save the JSON
    response_json = json.loads(json_content)
    
    # Save the JSON content
    with open(output_path, 'w', encoding='utf-8') as json_file:
        json.dump(response_json, json_file, ensure_ascii=False, indent=4)
    
    # Save the thinking part
    with open(thinking_path, 'w', encoding='utf-8') as thinking_file:
        thinking_file.write(thinking_part)
    
    return response_json, thinking_part
    
def generate_outputs(pipe, data_list, model_id, batch_size=1):
    total_samples = len(data_list)
    total_batches = (total_samples + batch_size - 1) // batch_size

    pbar = tqdm(total=total_batches, desc="Generating Outputs", unit="batch", leave=True)

    for i in range(0, total_samples, batch_size):
        batch_data = data_list[i:i+batch_size]
        batch_prompts = [item['prompt'] for item in batch_data]

        try:
            # Generate outputs for the current batch
            generated_outputs_batch = pipe(
                batch_prompts,
                max_new_tokens=6000,
                num_return_sequences=1,
                pad_token_id=pipe.tokenizer.eos_token_id,
                batch_size=batch_size,
                padding=True,
                return_full_text=False,
                temperature=0.1
            )

            # Process the generated outputs
            for idx, (generated_output, item) in enumerate(zip(generated_outputs_batch, batch_data)):
                output_path = item['output_path']
                error_file_path = item['error_file_path']
                relative_path = item['relative_path']
                file_id = item['file_id']
                assistant_suffix = item['assistant_suffix']
                
                # Create a path for thinking
                thinking_path = os.path.join(
                    os.path.dirname(output_path), 
                    f"{os.path.splitext(os.path.basename(output_path))[0]}_thinking.txt"
                )

                try:
                    if isinstance(generated_output, list) and len(generated_output) > 0:
                        generated_text = generated_output[0]['generated_text']
                    elif isinstance(generated_output, dict):
                        generated_text = generated_output['generated_text']
                    else:
                        raise ValueError(f"Unexpected generated_output structure: {type(generated_output)}")

                    # For models that return a dictionary with 'role' and 'content'
                    if isinstance(generated_text, dict) and 'content' in generated_text:
                        generated_text = generated_text['content']
                    
                    # Remove any assistant suffix that may be attached to the generated text
                    if assistant_suffix and generated_text.endswith(assistant_suffix):
                        generated_text = generated_text[:-len(assistant_suffix)]
                    
                    # Process the model's response
                    try:
                        #logging.info(f"generated text \n\ {generated_text}\n\n")
                        process_model_response(generated_text, output_path, thinking_path)
                        logging.info(f"Successfully processed {relative_path}")
                    except json.JSONDecodeError as json_error:
                        with open(error_file_path, 'w', encoding='utf-8') as error_file:
                            error_file.write(f"JSON Decode Error:\n{str(json_error)}\n\nOriginal LLM Response:\n{generated_text}")
                        with open(error_log, 'a', encoding='utf-8') as log_file:
                            log_file.write(f"{datetime.now()}: JSON decode error processing {relative_path}: {str(json_error)}\n")
                    except Exception as e:
                        with open(error_file_path, 'w', encoding='utf-8') as error_file:
                            error_file.write(f"Error:\n{str(e)}\n\nOriginal LLM Response:\n{generated_text}")
                        with open(error_log, 'a', encoding='utf-8') as log_file:
                            log_file.write(f"{datetime.now()}: Error processing {relative_path}: {str(e)}\n")

                except json.JSONDecodeError as json_error:
                    with open(error_file_path, 'w', encoding='utf-8') as error_file:
                        error_file.write(f"JSON Decode Error:\n{str(json_error)}\n\nOriginal LLM Response:\n{generated_text}")
                    with open(error_log, 'a', encoding='utf-8') as log_file:
                        log_file.write(f"{datetime.now()}: JSON decode error processing {relative_path}: {str(json_error)}\n")
                except Exception as e:
                    with open(error_file_path, 'w', encoding='utf-8') as error_file:
                        error_file.write(f"Error:\n{str(e)}\n\nOriginal LLM Response:\n{generated_text}")
                    with open(error_log, 'a', encoding='utf-8') as log_file:
                        log_file.write(f"{datetime.now()}: Error processing {relative_path}: {str(e)}\n")

        except torch.cuda.OutOfMemoryError as e:
            logging.error(f"OutOfMemoryError during processing batch starting at index {i}: {e}")
            # Handle the files in batch_data as error files
            for item in batch_data:
                error_file_path = item['error_file_path']
                relative_path = item['relative_path']
                with open(error_file_path, 'w', encoding='utf-8') as error_file:
                    error_file.write(f"CUDA Out of Memory Error during processing.\n")
                with open(error_log, 'a', encoding='utf-8') as log_file:
                    log_file.write(f"{datetime.now()}: OutOfMemoryError processing {relative_path}: {str(e)}\n")
            # Attempt to clear CUDA cache
            torch.cuda.empty_cache()
            # Reinitialize the model
            logging.info("Attempting to reinitialize the model after OutOfMemoryError.")
            try:
                del pipe
                torch.cuda.empty_cache()
                pipe = initialize_model(model_id)
            except Exception as e:
                logging.error(f"Failed to reinitialize model after OutOfMemoryError: {e}")
                break  # Cannot proceed further
            continue  # Proceed to next batch

        except Exception as e:
            logging.error(f"Unexpected error during processing batch starting at index {i}: {e}")
            # Handle the files in batch_data as error files
            for item in batch_data:
                error_file_path = item['error_file_path']
                relative_path = item['relative_path']
                with open(error_file_path, 'w', encoding='utf-8') as error_file:
                    error_file.write(f"Error during processing: {str(e)}\n")
                with open(error_log, 'a', encoding='utf-8') as log_file:
                    log_file.write(f"{datetime.now()}: Error processing {relative_path}: {str(e)}\n")
            continue  # Proceed to next batch

        pbar.update(1)

    pbar.close()
    logging.info(f"Generated outputs for {len(data_list)} samples.")

# ============================
# 6. Main Execution Function
# ============================

def main(model_id=None):
    if model_id is None:
        # Default model if none provided
        model_id = "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"
        
    logging.info(f"Using model: {model_id}")
    logging.info(f"Detected model family: {ModelFamily.detect_family(model_id)}")
    
    logging.info("Collecting file paths...")
    file_paths = collect_file_paths(input_folder)
    logging.info(f"Found {len(file_paths)} files to process")
    
    logging.info("Constructing prompts...")
    data_list = prepare_prompts(file_paths, input_folder, output_folder, errors_folder, 
                               system_prompt, user_content, model_id)
    logging.info(f"Prepared {len(data_list)} prompts")

    if not data_list:
        logging.info("No new files to process.")
        return

    # Initialize model across GPUs
    pipe = initialize_model(model_id)
    
    # Process all data with the model
    batch_size = 1  # Start with a conservative batch size
    generate_outputs(pipe, data_list, model_id, batch_size)

    logging.info("Generation process completed.")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Extract dataset information from research papers')
    parser.add_argument('--model', type=str, default="deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
                        help='The model ID to use for extraction')
    parser.add_argument('--input_folder', type=str, default="data/txt_files/",
                        help='Folder containing input text files')
    parser.add_argument('--output_folder', type=str, default="data/llama_tuned/json_files/",
                        help='Folder to save output JSON files')
    args = parser.parse_args()
    
    # Update global variables if provided in arguments
    input_folder = args.input_folder
    output_folder = args.output_folder
    
    mp.set_start_method('spawn')  # Important for CUDA multiprocessing
    main(args.model)