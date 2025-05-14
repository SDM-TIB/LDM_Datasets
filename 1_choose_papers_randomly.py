import json
import random
from tqdm import tqdm

# Load the JSON data from the file
with open('data/papers.json', 'r') as file:
    data = json.load(file)

# Define the target areas and the number of papers to select
areas_of_interest = [
    "Natural Language Processing",
    "Graphs",
    "Computer Vision",
    "General",
    "Reinforcement Learning",
    "Sequential"
]
papers_per_area = 2000
random_papers_without_methods = 3000

# Dictionary to store selected papers by area
selected_papers = {area: [] for area in areas_of_interest}
papers_without_methods = []

# Separate papers into those with methods and those without
for paper in tqdm(data):
    methods = paper.get("methods", [])
    paper_areas = {method.get("main_collection", {}).get("area") for method in methods if method.get("main_collection")}
    
    # Check if paper matches any area of interest
    matched = False
    for area in areas_of_interest:
        if area in paper_areas and len(selected_papers[area]) < papers_per_area:
            selected_papers[area].append(paper)
            matched = True
            break
    
    # If no methods and no match, add to papers_without_methods
    if not methods and not matched:
        papers_without_methods.append(paper)

# Randomly sample 100 unique papers from each area
for area in tqdm(areas_of_interest):
    if len(selected_papers[area]) >= papers_per_area:
        selected_papers[area] = random.sample(selected_papers[area], papers_per_area)
    else:
        print(f"Warning: Not enough papers found for area {area}")

# Randomly sample 300 unique papers from those without methods
if len(papers_without_methods) >= random_papers_without_methods:
    random_papers = random.sample(papers_without_methods, random_papers_without_methods)
else:
    print("Warning: Not enough papers found without methods.")
    random_papers = papers_without_methods

# Combine all selected papers into one list
final_selection = []
for area in tqdm(areas_of_interest):
    final_selection.extend(selected_papers[area])
final_selection.extend(random_papers)

# Write selected papers to a new JSON file, keeping the original structure
with open('data/selected_papers.json', 'w') as outfile:
    json.dump(final_selection, outfile, indent=2)
