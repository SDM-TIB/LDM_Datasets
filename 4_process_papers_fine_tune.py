import os
import json
import asyncio
import logging
import re
import time
import io
from pathlib import Path
from typing import Dict, List, Any, TypedDict, Annotated, Optional
from concurrent.futures import ThreadPoolExecutor
from tqdm.asyncio import tqdm  # Import tqdm for asyncio
from tqdm import tqdm as std_tqdm

# Import required LangGraph libraries
from langgraph.graph import StateGraph, START, END
from openai import AsyncOpenAI
import nest_asyncio

# Apply nest_asyncio to allow nested event loops
nest_asyncio.apply()

# Configure main logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configure separate progress logging
progress_log_file = Path("data/pwc/fine_tune/progress.log")
progress_log_file.parent.mkdir(parents=True, exist_ok=True)

# Custom progress bar that writes to a file instead of stdout
class FileProgressBar(std_tqdm):
    """A custom progress bar that writes updates to a file."""
    
    def __init__(self, *args, **kwargs):
        self.file_path = kwargs.pop('file_path', None)
        if self.file_path:
            self.log_file = open(self.file_path, 'w', encoding='utf-8')
            kwargs['file'] = self.log_file
        
        # Call the parent constructor
        super().__init__(*args, **kwargs)
        
        # Write initial progress information
        if self.file_path:
            self.write_progress_header()
    
    def write_progress_header(self):
        """Write header information to the progress file."""
        self.log_file.write(f"Progress tracking started at: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        self.log_file.write(f"Total items to process: {self.total}\n")
        self.log_file.write("=" * 80 + "\n")
        self.log_file.flush()
    
    def update(self, n=1):
        """Update the progress bar and write status to file."""
        super().update(n)
        if hasattr(self, 'log_file'):
            self.log_file.write(f"[{time.strftime('%H:%M:%S')}] Processed: {self.n}/{self.total} "
                              f"({self.n/self.total*100:.1f}%) - {self._get_postfix_str()}\n")
            self.log_file.flush()
    
    def set_postfix(self, **kwargs):
        """Set postfix and write to log file."""
        super().set_postfix(**kwargs)
        if hasattr(self, 'log_file'):
            self.log_file.write(f"[{time.strftime('%H:%M:%S')}] Status update: {self._get_postfix_str()}\n")
            self.log_file.flush()
    
    def _get_postfix_str(self):
        """Get the postfix string in a readable format."""
        if not hasattr(self, '_postfix'):
            return ""
        return ', '.join(f"{k}={v}" for k, v in self._postfix.items())
    
    def close(self):
        """Close the progress bar and file."""
        if hasattr(self, 'log_file'):
            self.log_file.write(f"\nProcessing completed at: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            self.log_file.write("=" * 80 + "\n")
            self.log_file.close()
        super().close()

# Initialize OpenAI client
client = AsyncOpenAI(api_key="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")

# Define model
MODEL = "gpt-4o-mini"

# Define prompts as constants
PROMPT_1 = """Extract all datasets mentioned in the paper below. For each dataset, find:
1. The name of the dataset.
2. The reference number and full citation in the references section. The reference number is the number that represents the citation number of the dataset. It's always written after the first mention of the dataset in a paper (it can also be a name not a number like "Sakor etl 201". here is an example below:


we used the Penn Action Dataset [40] and the sub-JHMDB Dataset [14]. 

reference part:
For the Penn Action Dataset, the reference is [40], which is  
[40] W. Zhang, M. Zhu, and K. G. Derpanis. From actemes to
action: A strongly-supervised representation for detailed ac-
tion understanding. In ICCV, 2013. 2, 5


For the sub-JHMDB Dataset, the reference is [14], which is [14] H. Jhuang, J. Gall, S. Zufﬁ, C. Schmid, and M. J. Black.
Towards understanding action recognition. In ICCV, 2013.


###
Paper text:"""

PROMPT_2 = """
double check the reference numbers, are all of them correct ? if not provide the correct ones with the same response style. if all are correct then just repeat them again without additional message from your side
"""

PROMPT_3_TEMPLATE = """
considering  the paper text below here are the datasets mentioned in the paper with the corresponding references:
{last_response}

Extract from the following text of a research paper the datasets that the authors used or mentioned in the paper.
            Describe each dataset using the following vocabulary, write the response in JSON format:

            **dcterms:creator**: The entity responsible for creating the dataset. [Look for the list of authors from the dataset reference/citation. For reference style in numbers, use the reference number to locate the dataset (e.g., search "[32]" if 32 is the reference number for the dataset). If the style for references looks like this (corpus (Volske et al., 2017)), then search for the dataset in references using the author name "Volske".] (if not found, assign empty list, do NOT mix between the papers authors and the dataset authors unless this is really the case. A common mistake is to assign the creators of the datasets as the authors of the article even though the dataset is only used/mentioned in the article. Do NOT consider organizations as creators, creatores should be persons)

            **dcterms:description**: A textual description of the dataset. [based on how the authors of this paper used or described the dataset]

            **dcterms:title**: The name of the dataset.

            **dcterms:issued**: The date/year the dataset was published. [Look for the year in the dataset reference/citation] (if not found, assign empty string)

            **dcterms:language**: The language of the data in the dataset if the dataset content is a text. [based on what the authors described in this paper, otherwise assign empty string]

            **dcterms:identifier**: A unique identifier for the dataset (e.g., DOI or URL). [look for dataset doi or url in the dataset reference] (if not found, assign empty string)

            **dcat:theme**: The thematic category of the dataset. [based on the authors description in this paper. Use a list of categories if more than one category can be assigned, in case of one category use a list with only one category inside it. In case of no categories to be found assign empty list]

            **dcat:keyword**: Keywords or tags describing the dataset. [generate list of keywords based on the usage and description of the dataset in this paper]

            **dcat:landingPage**: A web page providing access to the dataset. [look for dataset doi or url in the dataset reference or if it's mentioned in this paper. Otherwise, assign empty string]

            **dcterms:hasVersion**: A specific version of the dataset. [If mentioned explicitly by the authors, or if there is a number in the name of the dataset that represents the version. Otherwise assign an empty string]

            **dcterms:format**: The format of the dataset (like Text, Image, Audio,...). [based on how the authors used or described the dataset. Otherwise, assign empty string]

            **mls:task**: the tasks for which the dataset can be used. [based on the tasks that this paper is tackling. Use a list of tasks for one or more tasks, or assign empty list]

            ###
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
            ###
            
            Return ONLY the json response without any additional messages or text.
            Return only datasets from the input paper. don't try to extract papers or other things. We just need datasets.
            
            
            ###
            Paper text:
"""

PROMPT_4 = """
considering the text of a research paper below, write a thinking message for a LLM which describe step by step how to extract datasets from a research paper that is similar to the example below, writing the full citation for each dataset is very very important. Don't include the json structure in the thinking. Here is an example of how the thinking should look like:

<think>
Alright, I need to extract the datasets mentioned in the paper titled "LSTM Pose Machines" by Yue Luo et al., and describe each dataset using the specified JSON format.

First, I'll begin by carefully reading through the **abstract, introduction, approach, and experiments sections** to identify whether any datasets are explicitly named or described. In the abstract, the authors mention evaluating their method on **two large-scale video pose estimation benchmarks**. This is a clear signal that datasets are involved.

Next, in **section 1 (Introduction)**, the authors state: "The experiments showed that our approach significantly outperformed current state-of-the-art methods on two large-scale video pose estimation benchmarks." This hints at the presence of named datasets, but I need to find their exact names and details.

In **section 4.1 (Datasets)**, the paper clearly specifies the two datasets used for evaluation:

1. **Penn Action Dataset**: Described as a large dataset containing 2,326 video clips. The dataset includes 13 annotated joints per frame, and the number of frames per clip varies.

2. **sub-JHMDB Dataset**: A subset of the full JHMDB dataset, used for pose estimation in videos. This subset contains 316 video clips with 11,200 frames, and all joints are visible in this subset. There are three standard train/test splits.

In section 4.3 (Experiments), the authors mention that they used the **Penn Action Dataset** and **sub-JHMDB Dataset** for their experiments. This confirms that these datasets are indeed the ones they utilized.

Now, I'll consult the **References section** to retrieve full citations for these datasets:

- For **Penn Action Dataset**, the citation is:
  > Wenzheng Zhang, Menglong Zhu, and Konstantinos G. Derpanis. *From Actemes to Action: A Strongly-Supervised Representation for Detailed Action Understanding*. In Proceedings of the IEEE International Conference on Computer Vision (ICCV), pages 2248–2255, 2013.

- For **JHMDB Dataset**, the citation is:
  > Haoqi Fan Jhuang, Juergen Gall, Silvia Zuffi, Cordelia Schmid, and Michael J. Black. *Towards Understanding Action Recognition*. In Proceedings of the IEEE International Conference on Computer Vision (ICCV), pages 3192–3199, 2013.

Even though the authors use a subset of JHMDB (i.e., sub-JHMDB), the original paper is still the authoritative citation.

Now, I will create entries for each dataset using the provided JSON schema below.

After all of this, I will compile the dataset entries into a single structured JSON output that is ready for review or downstream processing.
</think>
"""

# Define our state
class PaperState(TypedDict):
    paper_text: str
    paper_path: str
    filename: str
    topic: str
    extract_result: Optional[str]
    doublecheck_result: Optional[str]
    json_result: Optional[str]
    thinking_result: Optional[str]
    error: Optional[str]

# Helper functions
async def read_file(file_path: Path) -> str:
    """Read the content of a text file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()
    except Exception as e:
        logger.error(f"Error reading file {file_path}: {e}")
        return ""

async def save_file(file_path: Path, content: str) -> None:
    """Save content to a file."""
    try:
        # Create directories if they don't exist
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(file_path, 'w', encoding='utf-8') as file:
            file.write(content)
        logger.info(f"Saved file: {file_path}")
    except Exception as e:
        logger.error(f"Error saving file {file_path}: {e}")

async def openai_call(prompt: str, max_retries: int = 3, retry_delay: int = 5) -> Optional[str]:
    """Make an API call to OpenAI with retry logic."""
    for attempt in range(max_retries):
        try:
            response = await client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": "You are a helpful AI assistant."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                timeout=60  # 60 seconds timeout
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.warning(f"API call failed (attempt {attempt+1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                # Wait before retrying
                await asyncio.sleep(retry_delay * (attempt + 1))  # Exponential backoff
            else:
                logger.error(f"API call failed after {max_retries} attempts")
                return None

def extract_json(text: str) -> str:
    """Extract valid JSON from text response."""
    # Log the input for debugging
    logger.debug(f"Attempting to extract JSON from: {text[:100]}...")
    
    # First, check if the entire text is valid JSON
    try:
        json.loads(text)
        return text
    except json.JSONDecodeError:
        pass
    
    # Try to find JSON array within the text using a more robust pattern
    # Look for either array or object
    json_patterns = [
        r'\[\s*\{.*\}\s*\]',  # For JSON arrays
        r'\{\s*"[^"]+"\s*:.*\}'  # For JSON objects
    ]
    
    for pattern in json_patterns:
        matches = re.findall(pattern, text, re.DOTALL)
        for match in matches:
            try:
                # Validate it's proper JSON by parsing it
                json.loads(match)
                return match
            except json.JSONDecodeError:
                continue
    
    # If we can't extract valid JSON, return a fallback format
    logger.warning("Could not extract valid JSON from response, returning sanitized text")
    
    # Try to create a minimally valid JSON structure
    try:
        # Check if text starts with a curly brace or square bracket
        if re.search(r'^\s*[\{\[]', text):
            # Try to clean up and parse
            cleaned = re.sub(r'[^\x00-\x7F]+', '', text)  # Remove non-ASCII chars
            json.loads(cleaned)
            return cleaned
    except:
        pass
    
    # Last resort: return the text in a safe JSON format
    return json.dumps({"error": "Could not parse valid JSON", "raw_text": text})

def extract_thinking(text: str) -> str:
    """Extract the thinking content from between <think> tags."""
    pattern = r'<think>(.*?)</think>'
    match = re.search(pattern, text, re.DOTALL)
    
    if match:
        return match.group(1).strip()
    else:
        # If no tags found, check if response might already be just the thinking content
        if "<think>" not in text and "</think>" not in text:
            return text.strip()
        return ""

# Define the LangGraph nodes (steps)
async def extract_datasets(state: PaperState) -> PaperState:
    """First step: Extract datasets and references."""
    try:
        prompt = f"{PROMPT_1}{state['paper_text']}"
        response = await openai_call(prompt)
        
        if response:
            state["extract_result"] = response
            return state
        else:
            state["error"] = "Failed at extract_datasets step"
            return state
    except Exception as e:
        state["error"] = f"Error in extract_datasets: {str(e)}"
        logger.error(state["error"])
        return state

async def double_check_references(state: PaperState) -> PaperState:
    """Second step: Double check references."""
    if state.get("error"):
        return state
        
    try:
        prompt = f"{PROMPT_1}{state['paper_text']}\n\n{state['extract_result']}\n\n{PROMPT_2}"
        response = await openai_call(prompt)
        
        if response:
            state["doublecheck_result"] = response
            return state
        else:
            state["error"] = "Failed at double_check_references step"
            return state
    except Exception as e:
        state["error"] = f"Error in double_check_references: {str(e)}"
        logger.error(state["error"])
        return state

async def generate_json(state: PaperState) -> PaperState:
    """Third step: Generate JSON format."""
    if state.get("error"):
        return state
        
    try:
        # Create prompt by direct concatenation instead of using format()
        prompt_template = PROMPT_3_TEMPLATE.replace("{last_response}", state["doublecheck_result"])
        prompt = prompt_template + state["paper_text"]
        
        response = await openai_call(prompt)
        
        if response:
            state["json_result"] = response
            return state
        else:
            state["error"] = "Failed at generate_json step"
            return state
    except Exception as e:
        state["error"] = f"Error in generate_json: {str(e)}"
        logger.error(state["error"])
        return state

async def generate_thinking(state: PaperState) -> PaperState:
    """Fourth step: Generate thinking process."""
    if state.get("error"):
        return state
        
    try:
        prompt = f"{PROMPT_4}\n\n{state['paper_text']}"
        response = await openai_call(prompt)
        
        if response:
            thinking = extract_thinking(response)
            state["thinking_result"] = thinking
            return state
        else:
            state["error"] = "Failed at generate_thinking step"
            return state
    except Exception as e:
        state["error"] = f"Error in generate_thinking: {str(e)}"
        logger.error(state["error"])
        return state

def clean_json_response(text: str) -> str:
    """Remove Markdown code block formatting from JSON responses."""
    # Remove ```json at the beginning
    text = re.sub(r'^```json\s*', '', text)
    text = re.sub(r'^```\s*', '', text)  # Also try without 'json'
    
    # Remove ``` at the end
    text = re.sub(r'\s*```$', '', text)
    
    # Trim whitespace
    text = text.strip()
    
    return text

async def save_results(state: PaperState) -> PaperState:
    """Final step: Save results to files."""
    if state.get("error"):
        logger.error(f"Error processing {state['filename']}: {state['error']}")
        return state
    
    try:
        # Output files
        output_dir = Path("data/pwc/fine_tune/json_files") / state["topic"]
        json_file = output_dir / f"{Path(state['filename']).stem}.json"
        thinking_file = output_dir / f"{Path(state['filename']).stem}_thinking.txt"
        
        # Clean JSON response before saving
        cleaned_json = clean_json_response(state["json_result"])
        
        # Save JSON result
        await save_file(json_file, cleaned_json)
        
        # Save thinking file
        await save_file(thinking_file, state["thinking_result"])
        
        logger.info(f"Successfully processed {state['filename']}")
        return state
    except Exception as e:
        state["error"] = f"Error in save_results: {str(e)}"
        logger.error(state["error"])
        return state

def should_end(state: PaperState) -> bool:
    """Determine if we should end the flow."""
    return state.get("error") is not None

# Define the LangGraph
def build_graph():
    """Build the LangGraph workflow."""
    # Create a new graph
    graph = StateGraph(PaperState)
    
    # Add nodes
    graph.add_node("extract_datasets", extract_datasets)
    graph.add_node("double_check_references", double_check_references)
    graph.add_node("generate_json", generate_json)
    graph.add_node("generate_thinking", generate_thinking)
    graph.add_node("save_results", save_results)
    
    # Add entry point - THIS IS CRITICAL
    graph.add_edge(START, "extract_datasets")
    
    # Add edges
    graph.add_edge("extract_datasets", "double_check_references")
    graph.add_edge("double_check_references", "generate_json")
    graph.add_edge("generate_json", "generate_thinking")
    graph.add_edge("generate_thinking", "save_results")
    graph.add_edge("save_results", END)
    
    # Add conditional edges
    graph.add_conditional_edges(
        "extract_datasets",
        should_end,
        {
            True: END,
            False: "double_check_references"
        }
    )
    
    graph.add_conditional_edges(
        "double_check_references",
        should_end,
        {
            True: END,
            False: "generate_json"
        }
    )
    
    graph.add_conditional_edges(
        "generate_json",
        should_end,
        {
            True: END,
            False: "generate_thinking"
        }
    )
    
    graph.add_conditional_edges(
        "generate_thinking",
        should_end,
        {
            True: END,
            False: "save_results"
        }
    )
    
    return graph.compile()

# Function to process a single paper
async def process_paper(paper_path: Path, pbar=None) -> Dict[str, Any]:
    """Process a single paper through the LangGraph workflow."""
    paper_filename = paper_path.name
    topic = paper_path.parent.name
    
    # Check if output files already exist
    output_dir = Path("data/pwc/fine_tune/json_files") / topic
    json_file = output_dir / f"{Path(paper_path).stem}.json"
    thinking_file = output_dir / f"{Path(paper_path).stem}_thinking.txt"
    
    # If both files exist, skip processing
    if json_file.exists() and thinking_file.exists():
        logger.info(f"Skipping already processed paper: {paper_path}")
        # Update progress bar if provided
        if pbar:
            pbar.update(1)
        return {
            "success": True,
            "filename": paper_filename,
            "topic": topic,
            "json_path": str(json_file),
            "thinking_path": str(thinking_file),
            "skipped": True
        }
    
    logger.info(f"Processing paper: {paper_path}")
    
    try:
        # Read paper content
        paper_text = await read_file(paper_path)
        if not paper_text:
            if pbar:
                pbar.update(1)
            return {"success": False, "filename": paper_filename, "topic": topic, "error": "Failed to read paper"}
        
        # Initialize state
        initial_state: PaperState = {
            "paper_text": paper_text,
            "paper_path": str(paper_path),
            "filename": paper_filename,
            "topic": topic,
            "extract_result": None,
            "doublecheck_result": None,
            "json_result": None,
            "thinking_result": None,
            "error": None
        }
        
        # Create LangGraph
        graph = build_graph()
        
        # Execute graph
        result_state = await graph.ainvoke(initial_state)
        
        # Check for errors
        if result_state.get("error"):
            if pbar:
                pbar.update(1)
            return {
                "success": False,
                "filename": paper_filename,
                "topic": topic,
                "error": result_state["error"]
            }
        
        # Update progress bar if provided
        if pbar:
            pbar.update(1)
            
        return {
            "success": True,
            "filename": paper_filename,
            "topic": topic,
            "json_path": str(json_file),
            "thinking_path": str(thinking_file),
            "skipped": False
        }
    except Exception as e:
        logger.error(f"Error processing paper {paper_path}: {e}")
        if pbar:
            pbar.update(1)
        return {"success": False, "filename": paper_filename, "topic": topic, "error": str(e)}

# Function to process multiple papers in parallel
async def process_papers_batch(paper_paths: List[Path], pbar=None) -> List[Dict[str, Any]]:
    """Process a batch of papers concurrently."""
    tasks = [process_paper(paper_path, pbar) for paper_path in paper_paths]
    return await asyncio.gather(*tasks)

# Main function to process all papers with parallelization
async def process_all_papers(max_concurrent: int = 20) -> List[Dict[str, Any]]:
    """Process all papers in the directory structure with concurrency."""
    # Get all paper files
    base_dir = Path("data/pwc/fine_tune/txt_files")
    paper_files = []
    
    for topic_dir in base_dir.glob("*"):
        if topic_dir.is_dir():
            for paper_file in topic_dir.glob("*.txt"):
                paper_files.append(paper_file)
    
    logger.info(f"Found {len(paper_files)} papers to process")
    
    # Create batches of papers
    batch_size = max_concurrent
    batches = [paper_files[i:i + batch_size] for i in range(0, len(paper_files), batch_size)]
    
    # Create overall progress bar that writes to file
    with FileProgressBar(
            total=len(paper_files), 
            desc="Processing papers", 
            unit="paper",
            file_path=progress_log_file) as pbar:
        
        # Process batches
        all_results = []
        for i, batch in enumerate(batches):
            logger.info(f"Processing batch {i+1}/{len(batches)} ({len(batch)} papers)")
            batch_results = await process_papers_batch(batch, pbar)
            all_results.extend(batch_results)
            
            # Calculate and display progress statistics
            processed_so_far = (i + 1) * batch_size if i < len(batches) - 1 else len(paper_files)
            successful_so_far = len([r for r in all_results if r.get("success", False)])
            skipped_so_far = len([r for r in all_results if r.get("skipped", False)])
            failed_so_far = len([r for r in all_results if not r.get("success", False)])
            
            # Update progress bar log file with stats
            pbar.set_postfix(
                successful=successful_so_far, 
                skipped=skipped_so_far, 
                failed=failed_so_far,
                batch=f"{i+1}/{len(batches)}"
            )
            
            # Also log a summary after each batch for visibility in the main log
            if i % 5 == 0 or i == len(batches) - 1:  # Log every 5 batches or at the end
                logger.info(f"Progress: {pbar.n}/{pbar.total} papers processed " 
                          f"({pbar.n/pbar.total*100:.1f}%) - "
                          f"Success: {successful_so_far}, Skipped: {skipped_so_far}, Failed: {failed_so_far}")
    
    return all_results

async def main():
    """Main function to orchestrate the entire process."""
    # Process all papers
    logger.info("Starting to process all papers")
    logger.info(f"Progress will be logged to: {progress_log_file}")
    
    start_time = time.time()
    results = await process_all_papers()
    end_time = time.time()
    
    # Log results
    successful = [r for r in results if r.get("success", False)]
    failed = [r for r in results if not r.get("success", False)]
    skipped = [r for r in successful if r.get("skipped", False)]
    processed = [r for r in successful if not r.get("skipped", False)]
    
    time_taken = end_time - start_time
    hours, remainder = divmod(time_taken, 3600)
    minutes, seconds = divmod(remainder, 60)
    
    logger.info(f"Processing completed in {int(hours)}h {int(minutes)}m {seconds:.2f}s")
    logger.info(f"Success: {len(successful)}, Failed: {len(failed)}, Skipped: {len(skipped)}, Newly Processed: {len(processed)}")
    
    # Save results summary
    summary = {
        "total": len(results),
        "successful": len(successful),
        "failed": len(failed),
        "skipped": len(skipped),
        "newly_processed": len(processed),
        "processing_time_seconds": time_taken,
        "processing_time_formatted": f"{int(hours)}h {int(minutes)}m {seconds:.2f}s",
        "start_time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(start_time)),
        "end_time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(end_time)),
        "failed_details": failed
    }
    
    summary_path = Path("data/pwc/fine_tune/processing_summary.json")
    await save_file(summary_path, json.dumps(summary, indent=2))
    
    print(f"Processing completed: {len(successful)} successful ({len(skipped)} skipped), {len(failed)} failed")
    print(f"Summary saved to {summary_path}")
    print(f"Detailed progress log available at {progress_log_file}")

if __name__ == "__main__":
    asyncio.run(main())