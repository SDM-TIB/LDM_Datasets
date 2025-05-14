import os
import json
import csv
import re

# Define the paths
base_dir = 'data/1k'
json_dir = os.path.join(base_dir, 'json_files')
txt_dir = os.path.join(base_dir, 'txt_files')
csv_file_path = 'data/combined_papers_metadata_1000_None.csv'

# Prepare CSV headers (replace ':' with '_')
csv_headers = [
    'paper', 'dcterms_accessRights', 'dcterms_creator', 'dcterms_description', 
    'dcterms_title', 'dcterms_issued', 'dcterms_language', 'dcterms_identifier', 
    'dcat_theme', 'dcterms_type', 'dcat_keyword', 'dcat_landingPage', 
    'dcterms_hasVersion', 'dcterms_format', 'mls_task'
]

# Function to clean text
def clean_text(text):
    """Remove unsupported characters and ensure proper encoding."""
    if not text:
        return ""
    # Replace unsupported Unicode characters with a placeholder or remove them
    text = re.sub(r'[^\x20-\x7E]+', '', text)  # Remove non-ASCII characters
    # Replace excessive spaces with a single space
    return re.sub(r'\s+', ' ', text).strip()

# Function to format paper text
def format_paper_text(text):
    """Replace newlines with a special placeholder to preserve formatting."""
    if not text:
        return ""
    # Replace newlines with a special placeholder
    text = text.replace('\n', '{{NEWLINE}}').replace('\r', '')
    return clean_text(text)

# Function to write rows within a specific range
def generate_csv(start_index=0, max_rows=None):
    """Generate the CSV file with rows in the specified range."""
    with open(csv_file_path, mode='w', newline='', encoding='utf-8') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=csv_headers, quoting=csv.QUOTE_ALL, escapechar='\\')
        writer.writeheader()

        # Initialize row counter
        row_count = 0
        current_index = 0

        # Walk through each topic directory in the json_files directory
        for topic in os.listdir(json_dir):
            topic_json_path = os.path.join(json_dir, topic)
            topic_txt_path = os.path.join(txt_dir, topic)
            
            # Ensure both topic directories exist in json_files and txt_files
            if os.path.isdir(topic_json_path) and os.path.isdir(topic_txt_path):
                
                # Loop over each JSON file in the topic's json_files directory
                for json_filename in os.listdir(topic_json_path):
                    if json_filename.endswith('.json'):
                        json_path = os.path.join(topic_json_path, json_filename)
                        txt_filename = json_filename.replace('.json', '.txt')
                        txt_path = os.path.join(topic_txt_path, txt_filename)
                        
                        # Check if corresponding TXT file exists
                        if os.path.isfile(json_path) and os.path.isfile(txt_path):
                            
                            # Read the txt file content
                            with open(txt_path, 'r', encoding='utf-8') as txt_file:
                                paper_text = txt_file.read()
                                paper_text = format_paper_text(paper_text)  # Replace newlines with placeholders
                            
                            # Read the json file content
                            with open(json_path, 'r', encoding='utf-8') as json_file:
                                datasets = json.load(json_file)
                                
                                # Loop through each dataset entry in JSON
                                for dataset in datasets:
                                    if current_index < start_index:
                                        # Skip rows until reaching the start index
                                        current_index += 1
                                        continue

                                    if max_rows is not None and row_count >= max_rows:
                                        # Stop when max rows is reached
                                        return

                                    # Prepare a row dictionary for each dataset
                                    row = {
                                        'paper': paper_text,
                                        'dcterms_accessRights': clean_text(dataset.get('dcterms:accessRights', '')),
                                        'dcterms_creator': clean_text(', '.join(dataset.get('dcterms:creator', []))),
                                        'dcterms_description': clean_text(dataset.get('dcterms:description', '')),
                                        'dcterms_title': clean_text(dataset.get('dcterms:title', '')),
                                        'dcterms_issued': dataset.get('dcterms:issued', ''),
                                        'dcterms_language': clean_text(dataset.get('dcterms:language', '')),
                                        'dcterms_identifier': clean_text(dataset.get('dcterms:identifier', '')),
                                        'dcat_theme': clean_text(', '.join(dataset.get('dcat:theme', []))),
                                        'dcterms_type': clean_text(', '.join(dataset.get('dcterms:type', []))),
                                        'dcat_keyword': clean_text(', '.join(dataset.get('dcat:keyword', []))),
                                        'dcat_landingPage': clean_text(dataset.get('dcat:landingPage', '')),
                                        'dcterms_hasVersion': clean_text(dataset.get('dcterms:hasVersion', '')),
                                        'dcterms_format': clean_text(dataset.get('dcterms:format', '')),
                                        'mls_task': clean_text(', '.join(dataset.get('mls:task', [])))
                                    }
                                    
                                    # Write the row to the CSV file
                                    writer.writerow(row)

                                    # Update counters
                                    row_count += 1
                                    current_index += 1

# Generate CSV with specific range
generate_csv(start_index=1000, max_rows=None)  # Adjust start_index and max_rows as needed
