import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pdfminer.high_level import extract_text
from pdfminer.layout import LAParams
from tqdm import tqdm  # For progress bar

def convert_pdf_to_text(pdf_path, txt_path):
    """Converts a single PDF file to a text file."""
    laparams = LAParams()
    laparams.detect_vertical = False
    laparams.all_texts = True

    try:
        text = extract_text(pdf_path, laparams=laparams)
        with open(txt_path, 'w', encoding='utf-8') as txt_file:
            txt_file.write(text)
        return True
    except Exception as e:
        print(f"Failed to convert {pdf_path}. Error: {e}")
        return False

def prepare_and_convert_pdfs(src_root_dir, dest_root_dir):
    # List to store all PDF paths and their corresponding output paths
    tasks = []
    
    # Walk through all directories and subdirectories in the source directory
    for root, dirs, files in os.walk(src_root_dir):
        for filename in files:
            if filename.lower().endswith('.pdf'):
                pdf_path = os.path.join(root, filename)

                # Compute the relative path of the PDF file with respect to the source root directory
                rel_path = os.path.relpath(root, src_root_dir)

                # Create the corresponding directory in the destination root directory
                dest_dir = os.path.join(dest_root_dir, rel_path)
                os.makedirs(dest_dir, exist_ok=True)

                # Define the path for the output text file in the destination directory
                txt_filename = os.path.splitext(filename)[0] + '.txt'
                txt_path = os.path.join(dest_dir, txt_filename)

                # Check if the text file already exists to avoid redundant processing
                if not os.path.exists(txt_path):
                    # Add this file to the task list (PDF path and output text path)
                    tasks.append((pdf_path, txt_path))
                else:
                    print(f"Text file already exists for: {pdf_path}")

    # Set up a progress bar using tqdm
    total_tasks = len(tasks)
    with tqdm(total=total_tasks, desc="Converting PDFs", unit="file") as pbar:
        with ThreadPoolExecutor(max_workers=4) as executor:
            # Submit all tasks to the thread pool
            futures = {executor.submit(convert_pdf_to_text, pdf_path, txt_path): (pdf_path, txt_path) for pdf_path, txt_path in tasks}
            
            # Process each completed future
            for future in as_completed(futures):
                result = future.result()  # Check if the task succeeded
                if result:
                    pbar.update(1)  # Update the progress bar for each successful conversion
                else:
                    print(f"Conversion failed for: {futures[future][0]}")  # Log any failed conversions

if __name__ == "__main__":
    # Replace these paths with your actual source and destination root directories
    src_root_directory = 'data/pdfs'
    dest_root_directory = 'data/txt_files'

    if not os.path.exists(src_root_directory):
        print(f"The source directory {src_root_directory} does not exist.")
        sys.exit(1)

    # Create the destination root directory if it doesn't exist
    os.makedirs(dest_root_directory, exist_ok=True)

    # Run the preparation and conversion process
    prepare_and_convert_pdfs(src_root_directory, dest_root_directory)
