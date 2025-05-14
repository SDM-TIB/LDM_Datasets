import json
import os
import shutil
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import requests

# Load the selected papers
with open('data/papers.json', 'r') as file:
    selected_papers = json.load(file)

# Initialize the unique ID counter and a lock for thread safety
unique_id_counter = 1
unique_id_lock = threading.Lock()

# Define directories
base_pdf_dir = 'data/pdfs/'
arxiv_pdf_dir = 'data/arxiv_pdfs/'
os.makedirs(base_pdf_dir, exist_ok=True)

# Topics of interest
areas_of_interest = [
    "Natural Language Processing",
    "Graphs",
    "Computer Vision",
    "General",
    "Reinforcement Learning",
    "Sequential"
]

# Dictionary to keep track of downloaded papers
downloaded_papers = {area: [] for area in areas_of_interest}
downloaded_papers["No Topic"] = []  # For papers without a topic

# Function to handle arXiv papers
def handle_arxiv_paper(url_pdf, folder_name):
    try:
        # Extract the arXiv ID
        arxiv_id = url_pdf.split('/')[-1].split('v')[0]
        arxiv_path = os.path.join(arxiv_pdf_dir, arxiv_id[:4], f"{arxiv_id}.pdf")

        # Verify the file exists in the arXiv directory
        if not os.path.exists(arxiv_path):
            print(f"ArXiv paper {arxiv_id} not found in {arxiv_path}. Skipping.")
            return False

        # Assign a unique ID in a thread-safe manner
        with unique_id_lock:
            global unique_id_counter
            unique_id = f"paper_{unique_id_counter}"
            unique_id_counter += 1

        # Define the final PDF path using the unique ID
        pdf_path = os.path.join(folder_name, f"{unique_id}.pdf")

        # Check if the file already exists
        if os.path.exists(pdf_path):
            print(f"File {pdf_path} already exists. Skipping.")
            return False

        # Copy the file to the destination
        shutil.copy(arxiv_path, pdf_path)
        return unique_id
    except Exception as e:
        print(f"Error handling arXiv paper {url_pdf}: {e}")
        return False

# Function to download or copy a single PDF
def process_pdf(paper, folder_name):
    url_pdf = paper["url_pdf"]
    try:
        if "arxiv.org/pdf" in url_pdf:
            # Handle arXiv paper
            unique_id = handle_arxiv_paper(url_pdf, folder_name)
            if unique_id:
                paper['unique_id'] = unique_id
                return True
        else:
            # Non-arXiv paper: Download
            response = requests.get(url_pdf)
            if response.status_code == 200:
                # Assign a unique ID in a thread-safe manner
                with unique_id_lock:
                    global unique_id_counter
                    paper['unique_id'] = f"paper_{unique_id_counter}"
                    unique_id_counter += 1
                    pdf_path = os.path.join(folder_name, f"{paper['unique_id']}.pdf")

                # Check if the file already exists
                if os.path.exists(pdf_path):
                    print(f"File {pdf_path} already exists. Skipping.")
                    return False

                # Temporarily save to a unique temporary filename
                temp_pdf_path = os.path.join(folder_name, f"temp_{threading.get_ident()}.pdf")
                with open(temp_pdf_path, 'wb') as f:
                    f.write(response.content)

                # Check file size
                if os.path.getsize(temp_pdf_path) < 10240:  # 10KB
                    print(f"Downloaded file is too small, discarding.")
                    os.remove(temp_pdf_path)
                    return False

                # Rename the temp file to the final filename
                os.rename(temp_pdf_path, pdf_path)
                return True
            else:
                print(f"Failed to download paper: Status {response.status_code}")
                return False
    except Exception as e:
        print(f"Error processing paper: {e}")
        return False

# Main function to process PDFs by topic
def process_by_topic():
    total_processed = 0
    lock = threading.Lock()

    # Process for each specified area
    for area in areas_of_interest:
        topic_papers = [
            p for p in selected_papers
            if p.get("methods")
            and any(
                method is not None
                and method.get("main_collection")
                and method["main_collection"].get("area") == area
                for method in p["methods"]
            )
        ]
        folder_path = os.path.join(base_pdf_dir, area.replace(" ", "_"))
        os.makedirs(folder_path, exist_ok=True)

        with ThreadPoolExecutor(max_workers=200) as executor:
            future_to_paper = {executor.submit(process_pdf, paper, folder_path): paper for paper in topic_papers}
            for future in tqdm(as_completed(future_to_paper), total=len(future_to_paper), desc=f"Processing {area}"):
                paper = future_to_paper[future]
                try:
                    result = future.result()
                    if result:
                        with lock:
                            downloaded_papers[area].append(paper)
                            total_processed += 1
                except Exception as e:
                    print(f"Error processing paper: {e}")

    # Handle papers without a topic (No Topic)
    no_topic_papers = [p for p in selected_papers if not p.get("methods")]
    folder_path = os.path.join(base_pdf_dir, "No_Topic")
    os.makedirs(folder_path, exist_ok=True)

    with ThreadPoolExecutor(max_workers=200) as executor:
        future_to_paper = {executor.submit(process_pdf, paper, folder_path): paper for paper in no_topic_papers}
        for future in tqdm(as_completed(future_to_paper), total=len(future_to_paper), desc="Processing No Topic"):
            paper = future_to_paper[future]
            try:
                result = future.result()
                if result:
                    with lock:
                        downloaded_papers["No Topic"].append(paper)
                        total_processed += 1
            except Exception as e:
                    print(f"Error processing paper: {e}")

# Save the processed metadata to a new JSON file
def save_processed_metadata():
    processed_papers_list = [paper for papers in downloaded_papers.values() for paper in papers]
    with open('data/downloaded_papers.json', 'w') as outfile:
        json.dump(processed_papers_list, outfile, indent=2)

# Run the processing and save metadata
process_by_topic()
save_processed_metadata()
