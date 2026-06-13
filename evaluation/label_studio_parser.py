import json
import os

LABEL_STUDIO_EXPORT = "label_studio_export.json"
OUTPUT_TRAIN_FILE = "./train.json"

TARGET_EMAIL = "ashton.elliott@utdallas.edu" 

def isolate_single_annotator(input_path, output_path, target_email):
    with open(input_path, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)
        
    flattened_dataset = []
    skipped_count = 0
    
    for entry in raw_data:
        # extract the source text information
        data_block = entry.get("data", {})
        source_article = data_block.get("source_article", data_block.get("text", ""))
        question = data_block.get("question", "What arms or methods of force are used?")
        task_id = entry.get("id")
        
        # look for the specific block belonging to the target email
        target_anno = None
        for anno in entry.get("annotations", []):
            if anno.get("completed_by", {}).get("email") == target_email:
                target_anno = anno
                break
        
        # if the target person hasn't annotated this specific article yet, skip it entirely
        if not target_anno:
            skipped_count += 1
            continue
            
        # build the flat dictionary row for this article
        flattened_entry = {
            "id": task_id,
            "source_article": source_article,
            "question": question,
            "answer_labels": []
        }
        
        # extract only their specific highlights
        results = target_anno.get("result", [])
        for res in results:
            value = res.get("value", {})
            if "start" in value and "end" in value:
                flattened_entry["answer_labels"].append({
                    "start": value.get("start"),
                    "end": value.get("end"),
                    "text": value.get("text"),
                    "labels": value.get("labels", ["Answer"])
                })
        
        # if this person flagged the article as containing no answers
        if len(flattened_entry["answer_labels"]) == 0:
            del flattened_entry["answer_labels"]
            flattened_entry["no_answer"] = "No arms or methods mentioned"
            
        flattened_dataset.append(flattened_entry)
        
    # write the filtered data to train.json
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(flattened_dataset, f, indent=4, ensure_ascii=False)
        
    print(f"Complete! Isolated annotations for: {target_email}")
    print(f"Created {len(flattened_dataset)} testing rows in train.json (Skipped {skipped_count} articles they hadn't touched).")

# execute the parser
isolate_single_annotator(LABEL_STUDIO_EXPORT, OUTPUT_TRAIN_FILE, TARGET_EMAIL)