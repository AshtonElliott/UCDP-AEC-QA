import json
import os
from collections import Counter

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LABEL_STUDIO_EXPORT = os.path.join(BASE_DIR, "data", "raw_inputs", "label_studio.json")
OUTPUT_FILE = os.path.join(BASE_DIR, "data", "raw_inputs", "ground_truth.json")
CONFLICTS_FILE = os.path.join(BASE_DIR, "data", "raw_inputs", "annotator_conflicts.json")

TEXT_FIELD = "source_article"
QUESTION_FIELD = "question"

def build_scalable_consensus_dataset(input_path, output_path, conflicts_path):
    with open(input_path, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)
        
    consensus_dataset = []
    conflicts_dataset = [] 
    skipped_count = 0
    conflict_count = 0
    
    for entry in raw_data:
        data_block = entry.get("data", {})
        task_id = entry.get("id")
        
        annotations = entry.get("annotations", [])
        num_annotators = len(annotations)
        
        # skip if nobody annotated it
        if num_annotators == 0:
            skipped_count += 1
            continue
            
        annotator_freqs = []
        no_answer_votes = 0
        
        # gather spans and labels for each annotator
        for anno in annotations:
            results = anno.get("result", [])
            spans = []
            
            for res in results:
                val = res.get("value", {})
                text = val.get("text", "").strip().lower()
                
                # grab the labels
                labels = tuple(val.get("labels", []))
                
                # only add if it's an actual highlighted text span
                if text:
                    spans.append((text, labels))
            
            # if they highlighted 0 words, they voted "No Answer"
            if len(spans) == 0:
                no_answer_votes += 1
                
            annotator_freqs.append(Counter(spans))
                
        required_votes = (num_annotators // 2) + 1
        final_answer_labels = []
        
        # recreate a flattened entry with the task ID and data block
        flattened_entry = {"id": task_id}
        flattened_entry.update(data_block)
        
        # handle "No Answer"
        if no_answer_votes >= required_votes:
            flattened_entry["no_answer"] = True
            consensus_dataset.append(flattened_entry)
            continue
            
        # calculate text + label frequency consensus
        all_unique_spans = set()
        for freq_dict in annotator_freqs:
            all_unique_spans.update(freq_dict.keys())
            
        for span_key in all_unique_spans:
            text, labels = span_key
            
            # count how many annotators found this exact string + label 
            counts = [freq_dict.get(span_key, 0) for freq_dict in annotator_freqs]
            counts.sort(reverse=True)
            
            # get the count that meets the majority threshold
            consensus_count = counts[required_votes - 1]
            
            for _ in range(consensus_count):
                final_answer_labels.append({
                    "text": text,
                    "labels": list(labels)
                })
                
        # finalize the entry
        if len(final_answer_labels) > 0:
            flattened_entry["answer_labels"] = final_answer_labels
            # route agreed articles to the consensus list
            consensus_dataset.append(flattened_entry) 
        else:
            # 'No Answer' if there was total disagreement
            flattened_entry["no_answer"] = True
            flattened_entry["conflict_flag"] = True
            conflict_count += 1
            
            # add exactly what people voted for directly into the conflict JSON
            flattened_entry["annotator_claims"] = [
                {f"annotator_{i+1}": [f"'{t}' {list(l)} ({c}x)" for (t, l), c in fd.items()] if fd else "Voted No Answer"}
                for i, fd in enumerate(annotator_freqs)
            ]
            
            # route the disagreed articles to the conflict list
            conflicts_dataset.append(flattened_entry) 
            
            # print disagreement details for debugging
            print("\n" + "!" * 80)
            print(f"CONFLICT DETECTED: Task ID {task_id}")
            print("-" * 80)
            
            # grab the text 
            article_text = data_block.get(TEXT_FIELD, data_block.get("text", str(data_block)))
            print(f"SOURCE TEXT:\n{article_text}\n")
            
            print("ANNOTATOR SPANS:")
            for i, freq_dict in enumerate(annotator_freqs):
                if not freq_dict:
                    print(f"  Annotator {i+1}: Voted 'No Answer'")
                else:
                    print(f"  Annotator {i+1}:")
                    for (span_text, span_labels), count in freq_dict.items():
                        print(f"    - '{span_text}' {list(span_labels)} (Found {count}x)")
            print("!" * 80 + "\n")
            
    # export both files
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # save clean dataset
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(consensus_dataset, f, indent=4, ensure_ascii=False)
        
    # save the conflicts
    with open(conflicts_path, 'w', encoding='utf-8') as f:
        json.dump(conflicts_dataset, f, indent=4, ensure_ascii=False)
        
    print("=" * 60)
    print("Consensus Generation Complete!")
    print(f"Total clean articles in ground_truth.json: {len(consensus_dataset)}")
    print(f"Articles skipped (unannotated): {skipped_count}")
    print(f"Articles exported to annotator_conflicts.json: {conflict_count}")
    print("=" * 60)

if __name__ == "__main__":
    build_scalable_consensus_dataset(LABEL_STUDIO_EXPORT, OUTPUT_FILE, CONFLICTS_FILE)