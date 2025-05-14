import pandas as pd

def reorder_columns(csv_file, output_file):
    # 1. Load the CSV file
    df = pd.read_csv(csv_file)
    all_cols = df.columns.tolist()

    # 2. Static columns always at the front
    static_cols = [
        'HITId',
        'WorkerId',
        'AssignmentStatus',
        'WorkTimeInSeconds',
        'LifetimeApprovalRate',
        'Input.paper'
    ]

    # 3. Define the exact field order and their mappings
    #    (original, corrected, true_col, false_col)
    field_mappings = [
        ('Input.dcterms_accessRights', 'Answer.corrected_accessRights',
         'Answer.accessRights_correct.True', 'Answer.accessRights_correct.False'),
        ('Input.dcterms_creator', 'Answer.corrected_creator',
         'Answer.creator_correct.True', 'Answer.creator_correct.False'),
        ('Input.dcterms_issued', 'Answer.corrected_date',
         'Answer.date_correct.True', 'Answer.date_correct.False'),
        ('Input.dcterms_description', 'Answer.corrected_description',
         'Answer.description_correct.True', 'Answer.description_correct.False'),
        ('Input.dcterms_format', 'Answer.corrected_format',
         'Answer.format_correct.True', 'Answer.format_correct.False'),
        ('Input.dcterms_identifier', 'Answer.corrected_identifier',
         'Answer.identifier_correct.True', 'Answer.identifier_correct.False'),
        ('Input.dcat_keyword', 'Answer.corrected_keywords',
         'Answer.keywords_correct.True', 'Answer.keywords_correct.False'),
        ('Input.dcat_landingPage', 'Answer.corrected_landingPage',
         'Answer.landingPage_correct.True', 'Answer.landingPage_correct.False'),
        ('Input.dcterms_language', 'Answer.corrected_language',
         'Answer.language_correct.True', 'Answer.language_correct.False'),
        ('Input.mls_task', 'Answer.corrected_task',
         'Answer.task_correct.True', 'Answer.task_correct.False'),
        ('Input.dcat_theme', 'Answer.corrected_theme',
         'Answer.theme_correct.True', 'Answer.theme_correct.False'),
        ('Input.dcterms_title', 'Answer.corrected_title',
         'Answer.title_correct.True', 'Answer.title_correct.False'),
        ('Input.dcterms_type', 'Answer.corrected_type',
         'Answer.type_correct.True', 'Answer.type_correct.False'),
        ('Input.dcterms_hasVersion', 'Answer.corrected_version',
         'Answer.version_correct.True', 'Answer.version_correct.False'),
    ]

    # 4. Start building the new column order
    new_order = []

    # 4a. Add static columns if present
    for col in static_cols:
        if col in all_cols:
            new_order.append(col)

    # 4b. Add each field group in the specified order
    for (original, corrected, true_col, false_col) in field_mappings:
        if original in all_cols:
            new_order.append(original)
        if corrected in all_cols:
            new_order.append(corrected)
        if true_col in all_cols:
            new_order.append(true_col)
        if false_col in all_cols:
            new_order.append(false_col)

    # 4c. After all annotation fields, add Answer.comments if it exists
    if 'Answer.comments' in all_cols:
        new_order.append('Answer.comments')

    # 5. Finally, include any remaining columns that are not yet in new_order
    remaining_cols = [c for c in all_cols if c not in new_order]
    new_order.extend(remaining_cols)

    # 6. Reorder the DataFrame and save
    df_reordered = df[new_order]
    df_reordered.to_csv(output_file, index=False)

    print(f"Reordered CSV saved as {output_file}")

if __name__ == "__main__":
    # Example usage
    input_files = [
        "data/aws/Batch_5281193_batch_results.csv",
        "data/aws/Batch_5282286_batch_results.csv"
    ]
    output_files = [
        "data/aws/Batch_5281193_reordered.csv",
        "data/aws/Batch_5282286_reordered.csv"
    ]

    for in_file, out_file in zip(input_files, output_files):
        reorder_columns(in_file, out_file)
