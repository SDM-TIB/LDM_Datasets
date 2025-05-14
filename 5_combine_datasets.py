import os
import json
from tqdm import tqdm  # Import tqdm for progress bars

def load_json_file(file_path):
    """Load a JSON file and return its contents."""
    with open(file_path, 'r', encoding='utf-8') as file:
        return json.load(file)

def save_to_json(data, file_path):
    """Save data to a JSON file."""
    with open(file_path, 'w', encoding='utf-8') as file:
        json.dump(data, file, indent=4, ensure_ascii=False)

def save_to_jsonl(data, file_path):
    """Save data to a JSONL file."""
    with open(file_path, 'w', encoding='utf-8') as file:
        for entry in data:
            json.dump(entry, file, ensure_ascii=False)
            file.write('\n')

def build_arxiv_id_to_doi_mapping(arxiv_metadata_file):
    """Build a mapping from arXiv id to DOI."""
    arxiv_id_to_doi = {}
    print("Building arXiv ID to DOI mapping...")
    with open(arxiv_metadata_file, 'r', encoding='utf-8') as file:
        total_lines = sum(1 for _ in open(arxiv_metadata_file, 'r', encoding='utf-8'))  # Get total lines for progress bar
        file.seek(0)  # Reset file pointer to the beginning
        for line in tqdm(file, total=total_lines, desc="Processing arXiv metadata"):
            try:
                record = json.loads(line)
                arxiv_id = record.get('id')
                doi = record.get('doi')
                if arxiv_id:
                    if doi:
                        arxiv_id_to_doi[arxiv_id] = doi
                    else:
                        arxiv_id_to_doi[arxiv_id] = f"10.48550/arXiv.{arxiv_id}"
            except json.JSONDecodeError:
                print(f"Warning: Could not parse line in {arxiv_metadata_file}")
    return arxiv_id_to_doi

def load_papers(downloaded_papers_file):
    """Load papers from downloaded_papers.json into a dictionary."""
    print("Loading downloaded papers...")
    papers = {}
    papers_list = load_json_file(downloaded_papers_file)
    for paper in tqdm(papers_list, desc="Processing downloaded papers"):
        unique_id = paper.get("unique_id")
        if unique_id:
            papers[unique_id] = paper
    return papers

def collect_datasets(directory, papers, arxiv_id_to_doi):
    """Collect datasets from all JSON files in the directory."""
    datasets_by_title = {}
    print("Collecting datasets from JSON files...")
    # Get all JSON files in the directory
    json_files = []
    for root, _, files in os.walk(directory):
        for filename in files:
            if filename.endswith('.json'):
                json_files.append(os.path.join(root, filename))

    for file_path in tqdm(json_files, desc="Processing JSON files"):
        try:
            datasets = load_json_file(file_path)
            unique_id = os.path.splitext(os.path.basename(file_path))[0]  # Remove directory and .json extension

            # Ensure datasets is a list of dictionaries
            if isinstance(datasets, dict):
                datasets = [datasets]
            elif isinstance(datasets, list):
                if all(isinstance(item, dict) for item in datasets):
                    pass  # It's a list of dictionaries
                else:
                    print(f"Warning: List in {file_path} does not contain dictionaries.")
                    continue  # Skip this file
            else:
                print(f"Warning: Unexpected data format in {file_path}")
                continue  # Skip this file

            for dataset in datasets:
                if isinstance(dataset, dict):
                    try:
                        dataset['unique_id'] = unique_id  # Add unique_id to each dataset
                    except TypeError as e:
                        print(f"Error processing dataset in {file_path}: {e}")
                        print(f"Type of dataset: {type(dataset)}")
                        print(f"Dataset content: {dataset}")
                        continue  # Skip this dataset

                    # Step 5: Add DOI to dataset
                    paper = papers.get(unique_id)
                    if paper:
                        arxiv_id = paper.get('arxiv_id')
                        if arxiv_id:
                            doi = arxiv_id_to_doi.get(arxiv_id)
                            if doi:
                                dataset['datacite:isDescribedBy'] = doi
                        else:
                            # arxiv_id is null
                            url_abs = paper.get('url_abs', '')
                            if 'https://aclanthology.org' in url_abs:
                                # Extract the code after 'https://aclanthology.org/'
                                code = url_abs.split('https://aclanthology.org/')[-1]
                                if code:
                                    doi = '10.18653/v1/' + code
                                    dataset['datacite:isDescribedBy'] = doi

                    title = dataset.get("dcterms:title")
                    if title:
                        # Ensure title is a string
                        if isinstance(title, list):
                            # Join list elements into a single string
                            title = ' '.join(title)
                        else:
                            # Convert title to string (in case it's not)
                            title = str(title)

                        if title not in datasets_by_title:
                            datasets_by_title[title] = []
                        datasets_by_title[title].append(dataset)
                else:
                    print(f"Warning: Dataset in {file_path} is not a dictionary.")
        except json.JSONDecodeError:
            print(f"Warning: {file_path} is not a valid JSON file.")

    return datasets_by_title

def merge_datasets(datasets_list):
    """Merge a list of datasets into one dataset according to specified rules."""
    merged_dataset = {}

    # All titles should be the same, so we can just take the first one
    merged_dataset["dcterms:title"] = datasets_list[0].get("dcterms:title", "")
    merged_dataset['unique_id'] = datasets_list[0]['unique_id']  # Keep the unique_id

    # dcterms:accessRights -> use the value where we don't have empty string, otherwise assign empty string
    access_rights = [d.get("dcterms:accessRights", "") for d in datasets_list]
    non_empty_access_rights = [ar for ar in access_rights if ar]
    merged_dataset["dcterms:accessRights"] = non_empty_access_rights[0] if non_empty_access_rights else ""

    # dcterms:creator -> majority voting or first if all different
    creators_list = [tuple(d.get("dcterms:creator", [])) for d in datasets_list]
    creators_counts = {}
    for creators in creators_list:
        creators_counts[creators] = creators_counts.get(creators, 0) + 1
    max_count = max(creators_counts.values())
    majority_creators = [creators for creators, count in creators_counts.items() if count == max_count]
    merged_dataset["dcterms:creator"] = list(majority_creators[0])

    # dcterms:description -> consider the longest description
    descriptions = [d.get("dcterms:description", "") for d in datasets_list]
    longest_description = max(descriptions, key=len)
    merged_dataset["dcterms:description"] = longest_description

    # dcterms:issued -> majority voting for non-empty strings or first
    issued_list = []
    for d in datasets_list:
        issued = d.get("dcterms:issued", "")
        if issued:
            if isinstance(issued, list):
                issued_key = tuple(issued)
            else:
                issued_key = issued
            issued_list.append(issued_key)
    if issued_list:
        issued_counts = {}
        for issued in issued_list:
            issued_counts[issued] = issued_counts.get(issued, 0) + 1
        max_count = max(issued_counts.values())
        majority_issued = [issued for issued, count in issued_counts.items() if count == max_count]
        issued = majority_issued[0]
        if isinstance(issued, tuple):
            merged_dataset["dcterms:issued"] = list(issued)
        else:
            merged_dataset["dcterms:issued"] = issued
    else:
        merged_dataset["dcterms:issued"] = ""

    # dcterms:language -> majority voting for non-empty strings or first
    language_list = []
    for d in datasets_list:
        language = d.get("dcterms:language", "")
        if language:
            if isinstance(language, list):
                language_key = tuple(language)
            else:
                language_key = language
            language_list.append(language_key)
    if language_list:
        language_counts = {}
        for lang in language_list:
            language_counts[lang] = language_counts.get(lang, 0) + 1
        max_count = max(language_counts.values())
        majority_language = [lang for lang, count in language_counts.items() if count == max_count]
        lang = majority_language[0]
        if isinstance(lang, tuple):
            merged_dataset["dcterms:language"] = list(lang)
        else:
            merged_dataset["dcterms:language"] = lang
    else:
        merged_dataset["dcterms:language"] = ""

    # dcterms:identifier -> majority voting for non-empty strings or first with value
    identifier_list = []
    for d in datasets_list:
        identifier = d.get("dcterms:identifier", "")
        if identifier:
            if isinstance(identifier, list):
                identifier_key = tuple(identifier)
            else:
                identifier_key = identifier
            identifier_list.append(identifier_key)
    if identifier_list:
        identifier_counts = {}
        for identifier in identifier_list:
            identifier_counts[identifier] = identifier_counts.get(identifier, 0) + 1
        max_count = max(identifier_counts.values())
        majority_identifier = [identifier for identifier, count in identifier_counts.items() if count == max_count]
        identifier = majority_identifier[0]
        if isinstance(identifier, tuple):
            merged_dataset["dcterms:identifier"] = list(identifier)
        else:
            merged_dataset["dcterms:identifier"] = identifier
    else:
        merged_dataset["dcterms:identifier"] = ""

    # dcat:theme -> combine all unique values
    themes = []
    for d in datasets_list:
        themes.extend(d.get("dcat:theme", []))
    merged_dataset["dcat:theme"] = list(set(themes))

    # dcterms:type -> majority voting
    types_list = [tuple(d.get("dcterms:type", [])) for d in datasets_list]
    type_counts = {}
    for types in types_list:
        type_counts[types] = type_counts.get(types, 0) + 1
    max_count = max(type_counts.values())
    majority_types = [types for types, count in type_counts.items() if count == max_count]
    merged_dataset["dcterms:type"] = list(majority_types[0])

    # dcat:keyword -> combine all unique values
    keywords = []
    for d in datasets_list:
        keywords.extend(d.get("dcat:keyword", []))
    merged_dataset["dcat:keyword"] = list(set(keywords))

    # dcat:landingPage -> majority voting for non-empty strings or first with value
    landing_page_list = []
    for d in datasets_list:
        landing_page = d.get("dcat:landingPage", "")
        if landing_page:
            if isinstance(landing_page, list):
                landing_page_key = tuple(landing_page)
            else:
                landing_page_key = landing_page
            landing_page_list.append(landing_page_key)
    if landing_page_list:
        landing_page_counts = {}
        for landing_page in landing_page_list:
            landing_page_counts[landing_page] = landing_page_counts.get(landing_page, 0) + 1
        max_count = max(landing_page_counts.values())
        majority_landing_page = [landing_page for landing_page, count in landing_page_counts.items() if count == max_count]
        landing_page = majority_landing_page[0]
        if isinstance(landing_page, tuple):
            merged_dataset["dcat:landingPage"] = list(landing_page)
        else:
            merged_dataset["dcat:landingPage"] = landing_page
    else:
        merged_dataset["dcat:landingPage"] = ""

    # dcterms:hasVersion -> majority voting for non-empty strings or first with value
    has_version_list = []
    for d in datasets_list:
        has_version = d.get("dcterms:hasVersion", "")
        if has_version:
            if isinstance(has_version, list):
                has_version_key = tuple(has_version)
            else:
                has_version_key = has_version
            has_version_list.append(has_version_key)
    if has_version_list:
        has_version_counts = {}
        for version in has_version_list:
            has_version_counts[version] = has_version_counts.get(version, 0) + 1
        max_count = max(has_version_counts.values())
        majority_version = [version for version, count in has_version_counts.items() if count == max_count]
        version = majority_version[0]
        if isinstance(version, tuple):
            merged_dataset["dcterms:hasVersion"] = list(version)
        else:
            merged_dataset["dcterms:hasVersion"] = version
    else:
        merged_dataset["dcterms:hasVersion"] = ""

    # dcterms:format -> majority voting for non-empty strings or first with value
    format_list = []
    for d in datasets_list:
        fmt = d.get("dcterms:format", "")
        if fmt:
            if isinstance(fmt, list):
                fmt_key = tuple(fmt)
            else:
                fmt_key = fmt
            format_list.append(fmt_key)
    if format_list:
        format_counts = {}
        for fmt in format_list:
            format_counts[fmt] = format_counts.get(fmt, 0) + 1
        max_count = max(format_counts.values())
        majority_format = [fmt for fmt, count in format_counts.items() if count == max_count]
        fmt = majority_format[0]
        if isinstance(fmt, tuple):
            merged_dataset["dcterms:format"] = list(fmt)
        else:
            merged_dataset["dcterms:format"] = fmt
    else:
        merged_dataset["dcterms:format"] = ""

    # mls:task -> combine all unique values
    tasks = []
    for d in datasets_list:
        tasks.extend(d.get("mls:task", []))
    merged_dataset["mls:task"] = list(set(tasks))

    # Collect all DOIs from datasets and include them as a list
    dois = []
    for d in datasets_list:
        doi = d.get('datacite:isDescribedBy')
        if doi:
            if isinstance(doi, list):
                dois.extend(doi)
            else:
                dois.append(doi)
    dois = list(set(dois))
    if dois:
        merged_dataset['datacite:isDescribedBy'] = dois

    return merged_dataset

def main():
    input_directory = 'data/json_files'
    output_file = 'data/datasets.json'  # Change to 'data/datasets.jsonl' if needed

    # Step 4: Build arxiv id to doi mapping
    arxiv_metadata_file = 'data/archive/arxiv-metadata-oai-snapshot.json'
    arxiv_id_to_doi = build_arxiv_id_to_doi_mapping(arxiv_metadata_file)

    # Step 5: Load papers and build mapping
    downloaded_papers_file = 'data/downloaded_papers.json'
    papers = load_papers(downloaded_papers_file)

    # Step 1: Collect datasets into datasets_by_title with all instances
    datasets_by_title = collect_datasets(input_directory, papers, arxiv_id_to_doi)

    # Step 2: Merge datasets according to specified rules
    print("Merging datasets...")
    merged_datasets = []
    for title, datasets_list in tqdm(datasets_by_title.items(), desc="Merging datasets"):
        merged_dataset = merge_datasets(datasets_list)
        # Step 3: Ignore datasets where "Dataset" is not in "dcterms:type"
        if "Dataset" in merged_dataset.get("dcterms:type", []):
            merged_datasets.append(merged_dataset)

    # Remove 'unique_id' from datasets before saving, if not needed
    for dataset in merged_datasets:
        dataset.pop('unique_id', None)

    # Choose the output format
    print(f"Saving combined dataset to {output_file}...")
    if output_file.endswith('.json'):
        save_to_json(merged_datasets, output_file)
    elif output_file.endswith('.jsonl'):
        save_to_jsonl(merged_datasets, output_file)

    print(f"Combined dataset saved to {output_file}")

if __name__ == "__main__":
    main()