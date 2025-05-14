#!/usr/bin/env python
"""
Preprocessing script for dataset extraction fine-tuning.
Now with text cleaning to remove unicode junk and (cid:XX) artifacts.
"""

import os
import json
import argparse
import random
import re
from typing import Dict, List, Tuple
from tqdm import tqdm
from datasets import Dataset
import numpy as np
from transformers import AutoTokenizer
from utils import get_prompt_template

def parse_args():
    parser = argparse.ArgumentParser(description="Preprocess dataset for fine-tuning")
    
    parser.add_argument("--txt_dir", type=str, required=True)
    parser.add_argument("--json_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="processed_data")
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-3.2-3B-Instruct")
    parser.add_argument("--train_ratio", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_seq_length", type=int, default=30000)
    
    return parser.parse_args()

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

def load_data(txt_dir: str, json_dir: str) -> List[Dict]:
    data = []
    topics = [d for d in os.listdir(txt_dir) if os.path.isdir(os.path.join(txt_dir, d))]
    
    for topic in topics:
        topic_txt_dir = os.path.join(txt_dir, topic)
        topic_json_dir = os.path.join(json_dir, topic)
        
        if not os.path.exists(topic_json_dir):
            print(f"Warning: JSON directory for topic {topic} not found, skipping")
            continue
            
        txt_files = [f for f in os.listdir(topic_txt_dir) if f.endswith('.txt')]
        
        for txt_file in tqdm(txt_files, desc=f"Processing {topic}"):
            paper_id = txt_file.replace('.txt', '')
            
            json_path = os.path.join(topic_json_dir, f"{paper_id}.json")
            thinking_path = os.path.join(topic_json_dir, f"{paper_id}_thinking.txt")
            
            if not (os.path.exists(json_path) and os.path.exists(thinking_path)):
                continue
                
            try:
                with open(os.path.join(topic_txt_dir, txt_file), 'r', encoding='utf-8') as f:
                    paper_text = f.read()
                    
                with open(json_path, 'r', encoding='utf-8') as f:
                    json_data = json.load(f)
                    json_str = json.dumps(json_data, indent=4,  ensure_ascii=False)
                    
                with open(thinking_path, 'r', encoding='utf-8') as f:
                    thinking_text = f.read()
                
                # CLEAN the texts
                paper_text = clean_text(paper_text)
                thinking_text = clean_text(thinking_text)
                
                data.append({
                    'paper_id': paper_id,
                    'topic': topic,
                    'paper_text': paper_text,
                    'json_data': json_str,
                    'thinking_text': thinking_text,
                })
            except Exception as e:
                print(f"Warning: error loading {paper_id}: {e}")
                continue
    
    print(f"Loaded {len(data)} valid paper-JSON-thinking triplets")
    return data

def create_prompt_completion_pairs(data: List[Dict], tokenizer, model_name: str, max_seq_length: int) -> List[Dict]:
    prompt_template_fn = get_prompt_template(model_name)
    formatted_data = []
    skipped_count = 0
    truncated_count = 0
    
    # Set maximum length to 50k tokens as requested
    max_seq_length = min(max_seq_length, 50000)
    
    for item in tqdm(data, desc="Creating prompt-completion pairs"):
        # First tokenize the completion to know its length
        completion = f"<think>\n{item['thinking_text']}\n</think>\n\n```json\n{item['json_data']}\n```"
        completion_tokens = tokenizer.encode(completion)
        completion_length = len(completion_tokens)
        
        # Calculate how much space we have for the prompt
        available_length = max_seq_length - completion_length
        
        if available_length <= 0:
            # Completion alone is too long, skip this example
            skipped_count += 1
            continue
        
        # Get the paper text and tokenize it
        paper_text = item['paper_text']
        paper_tokens = tokenizer.encode(paper_text)
        
        # Get template without paper text to calculate its token length
        template_base = prompt_template_fn("")
        template_tokens = tokenizer.encode(template_base)
        template_length = len(template_tokens)
        
        # Calculate available space for paper text
        available_for_paper = available_length - template_length
        
        if len(paper_tokens) > available_for_paper:
            # Paper is too long - truncate from the beginning, keeping the end
            truncated_paper_tokens = paper_tokens[-available_for_paper:]
            truncated_paper_text = tokenizer.decode(truncated_paper_tokens)
            paper_text = truncated_paper_text
            truncated_count += 1
        
        # Format the full prompt with possibly truncated paper
        prompt = prompt_template_fn(paper_text)
        
        # Verify the total length one more time
        total_tokens = len(tokenizer.encode(prompt)) + completion_length
        if total_tokens > max_seq_length:
            # This should rarely happen, but just to be safe
            skipped_count += 1
            continue
        
        formatted_data.append({
            'paper_id': item['paper_id'],
            'topic': item['topic'],
            'prompt': prompt,
            'completion': completion,
            'token_count': total_tokens,
            'prompt_token_count': len(tokenizer.encode(prompt)),
            'completion_token_count': completion_length
        })
    
    print(f"Created {len(formatted_data)} pairs, truncated {truncated_count}, skipped {skipped_count} over max length")
    return formatted_data

def split_data(data: List[Dict], train_ratio: float, seed: int) -> Tuple[List[Dict], List[Dict]]:
    papers = {}
    for item in data:
        papers.setdefault(item['paper_id'], []).append(item)
    
    paper_ids = list(papers.keys())
    random.seed(seed)
    random.shuffle(paper_ids)
    
    split_idx = int(len(paper_ids) * train_ratio)
    train_ids = paper_ids[:split_idx]
    val_ids = paper_ids[split_idx:]
    
    train_data = [item for pid in train_ids for item in papers[pid]]
    val_data = [item for pid in val_ids for item in papers[pid]]
    
    return train_data, val_data

def save_to_jsonl(data: List[Dict], path: str):
    with open(path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

def create_hf_dataset(data: List[Dict]) -> Dataset:
    return Dataset.from_list(data)

def get_dataset_statistics(data: List[Dict]) -> Dict:
    stats = {'count': len(data), 'token_counts': {'total': [], 'prompt': [], 'completion': []}}
    for item in data:
        stats['token_counts']['total'].append(item['token_count'])
        stats['token_counts']['prompt'].append(item['prompt_token_count'])
        stats['token_counts']['completion'].append(item['completion_token_count'])
    
    for key in stats['token_counts']:
        counts = stats['token_counts'][key]
        if counts:
            stats[f'avg_{key}_tokens'] = sum(counts) / len(counts)
            stats[f'max_{key}_tokens'] = max(counts)
            stats[f'p99_{key}_tokens'] = np.percentile(counts, 99)
    
    return stats

def main():
    args = parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Loading tokenizer from {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    
    data = load_data(args.txt_dir, args.json_dir)
    formatted_data = create_prompt_completion_pairs(data, tokenizer, args.model_name, args.max_seq_length)
    
    train_data, val_data = split_data(formatted_data, args.train_ratio, args.seed)
    
    print(f"Train: {len(train_data)} samples")
    print(f"Val: {len(val_data)} samples")
    
    save_to_jsonl(train_data, os.path.join(args.output_dir, "train.jsonl"))
    save_to_jsonl(val_data, os.path.join(args.output_dir, "val.jsonl"))
    
    create_hf_dataset(train_data).save_to_disk(os.path.join(args.output_dir, "train_dataset"))
    create_hf_dataset(val_data).save_to_disk(os.path.join(args.output_dir, "val_dataset"))
    
    stats = {
        'train': get_dataset_statistics(train_data),
        'val': get_dataset_statistics(val_data)
    }
    
    with open(os.path.join(args.output_dir, "dataset_stats.json"), 'w') as f:
        json.dump(stats, f, indent=2)
    
    with open(os.path.join(args.output_dir, "preprocessing_args.json"), 'w') as f:
        json.dump(vars(args), f, indent=2)
    
    print(f"Done! Saved processed data to {args.output_dir}")

if __name__ == "__main__":
    main()
