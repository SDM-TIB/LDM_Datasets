import json
import os
import requests
from tqdm import tqdm

# Load the selected papers
with open('data/selected_papers.json', 'r') as file:
    selected_papers = json.load(file)

# Define the directory to save PDF files and create folders
base_pdf_dir = 'data/pdfs/'
os.makedirs(base_pdf_dir, exist_ok=True)

# Topics and target download counts
areas_of_interest = [
    "Natural Language Processing",
    "Graphs",
    "Computer Vision",
    "General",
    "Reinforcement Learning",
    "Sequential"
]
#target_download_count_per_area = 100
#target_download_count_no_topic = 400

# Dictionary to keep track of downloaded papers
downloaded_papers = {area: [] for area in areas_of_interest}
downloaded_papers["No Topic"] = []  # For papers without a topic

# Unique ID counter for papers across all topics
unique_id_counter = 1

# Function to download a single PDF synchronously and check file size
def download_pdf(paper, folder_name):
    global unique_id_counter
    # Assign a unique ID if it doesn't already have one
    if 'unique_id' not in paper:
        paper['unique_id'] = f"paper_{unique_id_counter}"
        unique_id_counter += 1
    
    url_pdf = paper["url_pdf"]
    pdf_path = os.path.join(folder_name, f"{paper['unique_id']}.pdf")
    
    # Skip if file already exists
    if os.path.exists(pdf_path):
        print(f"File {paper['unique_id']}.pdf already exists, skipping.")
        return True

    try:
        headers = {
        'User-Agent': 'PostmanRuntime/7.42.0',
        'Accept': 'application/pdf',
    }
        response = requests.get(url_pdf, headers=headers, stream=True)
        if response.status_code == 200:
            with open(pdf_path, 'wb') as f:
                f.write(response.content)
            
            # Check file size
            if os.path.getsize(pdf_path) < 10240:  # 10KB
                print(f"File {paper['title']}.pdf is too small, removing.")
                #os.remove(pdf_path)  # Delete if size is below threshold
                #return False
            return True  # Download and size check successful
        else:
            print(f"Failed to download {paper['unique_id']}: Status {response.status_code}")
            return False
    except Exception as e:
        print(f"Error downloading {paper['unique_id']}: {e}")
        return False


# Main function to download PDFs by topic and track progress
def download_by_topic():
    total_downloaded = 0

    # Download for each specified area
    for area in areas_of_interest:
        # if total_downloaded >= 1000:
        #     break

        # Filter papers by area, ensuring methods and main_collection are valid
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
        topic_download_count = 0
        folder_path = os.path.join(base_pdf_dir, area.replace(" ", "_"))
        os.makedirs(folder_path, exist_ok=True)

        for paper in tqdm(topic_papers, desc=f"Downloading {area}"):
            # if topic_download_count >= target_download_count_per_area:
            #     break
            # Only count if download was successful and file size is valid
            if download_pdf(paper, folder_path):
                downloaded_papers[area].append(paper)
                topic_download_count += 1
                total_downloaded += 1
            # if total_downloaded >= 1000:
            #     break

    # Handle papers without a topic (No Topic)
    #if total_downloaded < 1000:
    no_topic_papers = [p for p in selected_papers if not p.get("methods")]
    folder_path = os.path.join(base_pdf_dir, "No_Topic")
    os.makedirs(folder_path, exist_ok=True)

    no_topic_download_count = 0
    for paper in tqdm(no_topic_papers, desc="Downloading No Topic"):
        # if no_topic_download_count >= target_download_count_no_topic:
        #     break
        # Only count if download was successful and file size is valid
        if download_pdf(paper, folder_path):
            downloaded_papers["No Topic"].append(paper)
            no_topic_download_count += 1
            total_downloaded += 1
        # if total_downloaded >= 1000:
        #     break


# Save the downloaded metadata to a new JSON file once completed
def save_downloaded_metadata():
    downloaded_papers_list = [paper for papers in downloaded_papers.values() for paper in papers]
    with open('data/downloaded_papers.json', 'w') as outfile:
        json.dump(downloaded_papers_list, outfile, indent=2)


# Run the download and save metadata
download_by_topic()
save_downloaded_metadata()