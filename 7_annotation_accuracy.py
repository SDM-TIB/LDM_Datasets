import pandas as pd
import matplotlib.pyplot as plt

def compute_annotation_accuracy(file1, file2):
    """
    Reads two CSVs with Amazon Mechanical Turk results, filters out 
    'Rejected' rows, then calculates (and returns) annotation accuracy 
    for each field across both files combined.
    """
    # 1. Read CSV files
    df1 = pd.read_csv(file1)
    df2 = pd.read_csv(file2)
    
    # 2. Concatenate them into a single DataFrame
    df = pd.concat([df1, df2], ignore_index=True)
    
    # 3. Keep only Approved assignments
    df_approved = df[df['AssignmentStatus'] == 'Approved'].copy()
    
    # 4. List of the fields we want to compute accuracy for
    fields = [
        'accessRights', 'creator', 'date', 'description', 'format',
        'identifier', 'keywords', 'landingPage', 'language', 'task',
        'theme', 'title', 'type', 'version'
    ]
    
    # We will store accuracy results: field -> accuracy
    results = {}
    
    # 5. Group by HITId
    grouped = df_approved.groupby('HITId')
    
    for field in fields:
        col_true = f'Answer.{field}_correct.True'
        col_false = f'Answer.{field}_correct.False'
        
        correct_count = 0
        total_count = 0
        
        for hit_id, group in grouped:
            total_count += 1
            
            # Convert True/False columns to numeric if they come as strings like "1"/"0"
            # Adjust logic if your CSV uses "True"/"False" or "Yes"/"No".
            any_false = (group[col_false].astype(int) == 1).any()
            all_true = (group[col_true].astype(int) == 1).all()
            
            # If any annotator says False, it's incorrect
            # Only if all annotators say True, we count it as correct
            if not any_false and all_true:
                correct_count += 1
        
        accuracy = correct_count / total_count if total_count > 0 else 0.0
        results[field] = accuracy
    
    return results

def plot_accuracy_bar_chart(results):
    """
    Takes a dictionary {field: accuracy} and shows a bar chart 
    of the accuracy results using matplotlib.
    """
    fields = list(results.keys())
    accuracies = list(results.values())
    
    # Convert accuracy to percentage
    accuracies_percent = [acc * 100 for acc in accuracies]
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(fields, accuracies_percent, color='skyblue', edgecolor='black')
    
    # Add text labels above each bar
    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2.0, 
            height, 
            f"{height:.1f}%", 
            ha='center', 
            va='bottom',
            fontsize=9
        )
    
    plt.xlabel('Field')
    plt.ylabel('Accuracy (%)')
    plt.title('Annotation Accuracy by Field')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    file1 = "data/aws/Batch_5281193_batch_results.csv"
    file2 = "data/aws/Batch_5282286_batch_results.csv"
    
    accuracies = compute_annotation_accuracy(file1, file2)
    
    # Print numeric results
    print("Accuracy Results (fields vs. accuracy):")
    for field, acc in accuracies.items():
        print(f"{field}: {acc:.2%}")
    
    # Show bar chart
    plot_accuracy_bar_chart(accuracies)
