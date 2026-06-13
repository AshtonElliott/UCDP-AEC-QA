import json
import os
import numpy as np
from scipy.optimize import linear_sum_assignment

EVAL_FILE = 'mistral_Results.json' 
IOU_THRESHOLD = 0.5 

# converts a list of span dictionaries into a set of unique character positions
def get_character_indices(spans):
    indices = set()
    for span in spans:
        start = span.get('start')
        end = span.get('end')
        if start is not None and end is not None:
            indices.update(range(start, end))
    return indices

# calculates strict Intersection over Union (IoU) between character sets
def calculate_iou(gold_set, pred_set):
    if not gold_set and not pred_set:
        return 1.0  # both correctly agreed there are no annotations
    if not gold_set or not pred_set:
        return 0.0  # conflict: one found a label, the other didn't
    
    intersection = len(gold_set.intersection(pred_set))
    union = len(gold_set.union(pred_set))
    return intersection / union

def main():
    # verify file exists before running
    if not os.path.exists(EVAL_FILE):
        print(f"Error: Missing evaluation file at {EVAL_FILE}")
        return

    print(f"Loading dataset from {EVAL_FILE}...")
    with open(EVAL_FILE, 'r', encoding='utf-8') as f:
        dataset = json.load(f)

    # global counts
    global_tp = 0
    global_fp = 0
    global_fn = 0
    global_character_iou = 0.0
    evaluated_count = 0

    print(f"STARTING EVALUATION RUN (IoU Threshold: {IOU_THRESHOLD})")

    for entry in dataset:
        entry_id = entry.get('id', 'unknown')
        
        # pull human annotations vs AI predictions directly from the same row object
        gold_spans = entry.get('answer_labels', [])
        pred_spans = entry.get('model_spans', []) 

        entry_tp = 0
        num_gold = len(gold_spans)
        num_pred = len(pred_spans)

        # optimal bipartite matching using Hungarian algorithm
        if num_gold > 0 and num_pred > 0:
            # build IoU matrix
            iou_matrix = np.zeros((num_gold, num_pred))
            for g_idx, g_span in enumerate(gold_spans):
                g_text = g_span.get('text', '').strip().lower()
                g_indices = get_character_indices([g_span])
                
                for p_idx, p_span in enumerate(pred_spans):
                    p_text = p_span.get('text', '').strip().lower()
                    p_indices = get_character_indices([p_span])
                    
                    # exact string match safety valve for repetitions
                    if g_text == p_text:
                        iou_matrix[g_idx, p_idx] = 1.0
                    else:
                        iou_matrix[g_idx, p_idx] = calculate_iou(g_indices, p_indices)
            
            # maximize assignment optimization matrix
            gold_ind, pred_ind = linear_sum_assignment(-iou_matrix)
            
            for g_idx, p_idx in zip(gold_ind, pred_ind):
                if iou_matrix[g_idx, p_idx] >= IOU_THRESHOLD:
                    entry_tp += 1

        entry_fp = num_pred - entry_tp
        entry_fn = num_gold - entry_tp

        # calculate standard full-text character IoU for the whole document
        whole_gold = get_character_indices(gold_spans)
        whole_pred = get_character_indices(pred_spans)
        doc_iou = calculate_iou(whole_gold, whole_pred)
        
        # accumulate metrics
        global_tp += entry_tp
        global_fp += entry_fp
        global_fn += entry_fn
        global_character_iou += doc_iou
        evaluated_count += 1

        print(f"Entry {evaluated_count} [ID: {entry_id}]:")
        print(f"  -> Human Spans:          {len(gold_spans)}")
        print(f"  -> AI Spans Sent:        {len(pred_spans)}")
        print(f"  -> Explicit Counts:      TP={entry_tp}, FP={entry_fp}, FN={entry_fn}")
        print(f"  -> Full Doc Text IoU:    {doc_iou:.4f}\n")

    # final calculations
    mean_iou = global_character_iou / evaluated_count if evaluated_count > 0 else 0
    
    # precision score: TP / (TP + FP)
    precision = global_tp / (global_tp + global_fp) if (global_tp + global_fp) > 0 else 0

    # recall score: TP / (TP + FN)
    recall = global_tp / (global_tp + global_fn) if (global_tp + global_fn) > 0 else 0

    # F1 score
    f1_score = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    print("="*60)
    print("FINAL METRICS SUMMARY")
    print("="*60)
    print(f"Total Unique Articles Audited:   {evaluated_count}")
    print(f"Mean Character-Level IoU:         {mean_iou:.4f}")
    print("-"*60)
    print("STRICT SPAN MATRIX ANALYSIS:")
    print(f"  -> Total True Positives (TP):   {global_tp}")
    print(f"  -> Total False Positives (FP):  {global_fp}")
    print(f"  -> Total False Negatives (FN):  {global_fn}")
    print("-"*60)
    print(f"  -> Final Precision Metric:      {precision:.4f}")
    print(f"  -> Final Recall Metric:         {recall:.4f}")
    print(f"  -> Final Balanced F1-Score:     {f1_score:.4f}")
    print("="*60 + "\n")

if __name__ == "__main__":
    main()