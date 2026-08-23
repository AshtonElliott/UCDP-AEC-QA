import json
import os
import argparse
import re
from collections import Counter

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) if '__file__' in locals() else os.getcwd()
RAW_DIR = os.path.join(BASE_DIR, "data", "raw_inputs")

TEXT_FIELD = "source_article"

def build_scalable_consensus_dataset(input_path, output_path, conflicts_path):
    if not os.path.exists(input_path):
        print(f"Error: Could not find input file at {input_path}")
        print("Please ensure your Label Studio export is named correctly.")
        return
        
    with open(input_path, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)
        
    consensus_dataset = []
    conflicts_dataset = [] 
    skipped_count = 0
    conflict_count = 0
    
    for entry in raw_data:
        data_block = entry.get("data", {})
        task_id = entry.get("id")
        
        raw_annotations = entry.get("annotations", [])
        
        # filter out cancelled annotations (e.g. if an annotator skipped it)
        valid_annotations = [a for a in raw_annotations if not a.get("was_cancelled", False)]
        num_annotators = len(valid_annotations)
        
        # skip if nobody annotated it yet
        if num_annotators == 0:
            skipped_count += 1
            continue
            
        annotator_freqs = []
        no_answer_votes = 0
        
        # gather spans and labels for each annotator
        for anno in valid_annotations:
            results = anno.get("result", [])
            spans = []
            voted_no_answer = False
            
            for res in results:
                # check if they explicitly clicked the "No Damage Detected" choice
                if res.get("type") == "choices" and res.get("from_name") == "no_answer":
                    voted_no_answer = True
                    
                # check if they highlighted a specific span
                elif res.get("type") == "labels":
                    val = res.get("value", {})
                    text = val.get("text", "").strip() # original casing preserved
                    labels = tuple(val.get("labels", []))
                    
                    if text:
                        spans.append((text, labels))
            
            # if they explicitly voted No Answer OR highlighted 0 words
            if voted_no_answer or len(spans) == 0:
                no_answer_votes += 1
                
            annotator_freqs.append(Counter(spans))
                
        required_votes = (num_annotators // 2) + 1
        final_answer_labels = []
        
        flattened_entry = {"id": task_id}
        flattened_entry.update(data_block)
        
        # handle majority "No Answer" consensus
        if no_answer_votes >= required_votes:
            flattened_entry["no_answer"] = "No Damage Detected"
            consensus_dataset.append(flattened_entry)
            continue
            
        # calculate text + label frequency consensus
        all_unique_spans = set()
        for freq_dict in annotator_freqs:
            all_unique_spans.update(freq_dict.keys())
            
        for span_key in all_unique_spans:
            text, labels = span_key
            
            counts = [freq_dict.get(span_key, 0) for freq_dict in annotator_freqs]
            counts.sort(reverse=True)
            consensus_count = counts[required_votes - 1]
            
            for _ in range(consensus_count):
                final_answer_labels.append({
                    "text": text,
                    "labels": list(labels)
                })
                
        if len(final_answer_labels) > 0:
            flattened_entry["answer_labels"] = final_answer_labels
            consensus_dataset.append(flattened_entry) 
        else:
            flattened_entry["no_answer"] = "Conflict / No Agreement"
            flattened_entry["conflict_flag"] = True
            conflict_count += 1
            
            flattened_entry["annotator_claims"] = [
                {f"annotator_{i+1}": [f"'{t}' {list(l)} ({c}x)" for (t, l), c in fd.items()] if fd else "Voted No Answer"}
                for i, fd in enumerate(annotator_freqs)
            ]
            
            conflicts_dataset.append(flattened_entry) 
            
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(consensus_dataset, f, indent=4, ensure_ascii=False)
        
    with open(conflicts_path, 'w', encoding='utf-8') as f:
        json.dump(conflicts_dataset, f, indent=4, ensure_ascii=False)
        
    print(f"Consensus Generation Complete for: {os.path.basename(output_path)}")
    print(f"  -> Total clean articles: {len(consensus_dataset)}")
    print(f"  -> Articles skipped (unannotated/cancelled): {skipped_count}")
    print(f"  -> Articles exported to conflicts file: {conflict_count}")
    print("-" * 60)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Consensus Ground Truth from Label Studio")
    parser.add_argument('--q', type=int, default=None, help="Question number to process (e.g., 1, 2)")
    
    args = parser.parse_args()
    
    if not os.path.exists(RAW_DIR):
        print(f"[ERROR] Raw directory not found at: {RAW_DIR}")
        sys.exit(1)
        
    qs_to_process = []
    
    if args.q is not None:
        qs_to_process.append(args.q)
    else:
        # Dynamically scan the directory for all label_studio_qX.json files
        for filename in os.listdir(RAW_DIR):
            match = re.match(r"label_studio_q(\d+)\.json", filename, re.IGNORECASE)
            if match:
                qs_to_process.append(int(match.group(1)))
                
        if not qs_to_process:
            print(f"[ERROR] Could not find any files matching 'label_studio_qX.json' in {RAW_DIR}")
            sys.exit(1)
            
    # Sort them so they process in order (1, 2, 3...)
    qs_to_process = sorted(qs_to_process)
    
    print("=" * 60)
    print(f"Starting Consensus Generation for Questions: {qs_to_process}")
    print("=" * 60)
    
    for q in qs_to_process:
        input_file = os.path.join(RAW_DIR, f"label_studio_q{q}.json")
        output_file = os.path.join(RAW_DIR, f"ground_truth_q{q}.json")
        conflicts_file = os.path.join(RAW_DIR, f"annotator_conflicts_q{q}.json")
        
        build_scalable_consensus_dataset(input_file, output_file, conflicts_file)