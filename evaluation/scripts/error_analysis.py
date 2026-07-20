import os
import sys
import json
from pathlib import Path
from collections import defaultdict

BASE_DIR = str(Path(__file__).resolve().parent.parent)
RESULTS_DIR = os.path.join(BASE_DIR, 'data', 'evaluation_results')
ARTIFACT_PATH = os.path.join(RESULTS_DIR, "master_evaluation_artifact.json")

def run_error_analysis(records=None):
    print("\n## Part 4: Error Analysis & Edge Cases", file=sys.stdout)
    print("> *Automated extraction of specific failure modes across the pipeline.*", file=sys.stdout)
    
    if records is None:
        if not os.path.exists(ARTIFACT_PATH):
            return
        with open(ARTIFACT_PATH, 'r', encoding='utf-8') as f:
            records = json.load(f)
            
    # structure: diagnostics[question_id]["error_type"] = [list of cases]
    diagnostics = defaultdict(lambda: {"paraphrase": [], "hallucination": [], "label_crash": []})

    for rec in records:
        q_num = rec.get('question', 1)
        g_text_str = " | ".join(rec['g_texts']) if rec['g_texts'] else "[NO TARGET SPANS EXIST]"
        p_text_str = " | ".join(rec['p_texts']) if rec['p_texts'] else "[NO SPANS PREDICTED]"

        # 1. Valid Paraphrasing (High Semantic score, Zero Exact Match)
        if rec['em'] == 0.0 and rec['dedup_f1'] > 0.70 and rec['g_texts'] and len(diagnostics[q_num]["paraphrase"]) < 3:
            diagnostics[q_num]["paraphrase"].append({
                "model": rec['model'], "gt": g_text_str, "pred": p_text_str, 
                "em": rec['em'], "f1": rec['dedup_f1']
            })

        # 2. Over-Extraction (Generated too much noise, low accuracy)
        elif rec['spans_generated'] >= 5 and rec['dedup_f1'] < 0.4 and len(diagnostics[q_num]["hallucination"]) < 3:
            diagnostics[q_num]["hallucination"].append({
                "model": rec['model'], "gt": g_text_str, "pred": p_text_str, 
                "spans": rec['spans_generated'], "f1": rec['dedup_f1']
            })

        # 3. Label Mismatch (Perfect text extraction, but mapped to the wrong category)
        elif rec['em'] > 0.8 and rec['strict_f1'] < 0.3 and rec['g_texts'] and len(diagnostics[q_num]["label_crash"]) < 3:
            g_lbl = [l for l in rec.get('g_labels', []) if l.strip("[]'\" ") != ""]
            p_lbl = [l for l in rec.get('p_labels', []) if l.strip("[]'\" ") != ""]
            
            g_lbl_clean = " | ".join(g_lbl) if g_lbl else "[NO LABEL]"
            p_lbl_clean = " | ".join(p_lbl) if p_lbl else "[NO LABEL]"
            
            if g_lbl_clean != p_lbl_clean:
                diagnostics[q_num]["label_crash"].append({
                    "model": rec['model'], "gt_text": g_text_str, "pred_text": p_text_str,
                    "gt_label": g_lbl_clean, "pred_label": p_lbl_clean
                })

    for q_num in sorted(diagnostics.keys()):
        print(f"\n### 🔍 Analysis: Question {q_num}")
        q_data = diagnostics[q_num]
        
        if q_data["paraphrase"]:
            print("\n#### A. Valid Paraphrasing (High Semantic Match, Zero Exact Match)")
            for i, case in enumerate(q_data["paraphrase"], 1):
                print(f"**Edge Case #{i} ({case['model']})**\n* **Ground Truth:** `{case['gt']}`\n* **Model Prediction:** `{case['pred']}`\n* **Scores:** Exact Match = {case['em']:.4f} | Semantic F1 = {case['f1']:.4f}\n---")

        if q_data["hallucination"]:
            print("\n#### B. Over-Extraction (High Verbosity, Low Precision)")
            for i, case in enumerate(q_data["hallucination"], 1):
                print(f"**Edge Case #{i} ({case['model']})**\n* **Ground Truth:** `{case['gt']}`\n* **Model Prediction:** `{case['pred']}`\n* **Scores:** Spans Generated = {case['spans']} | Semantic F1 = {case['f1']:.4f}\n---")

        if q_data["label_crash"]:
            print("\n#### C. Label Mismatch (Correct Text, Wrong Category)")
            for i, case in enumerate(q_data["label_crash"], 1):
                print(f"**Edge Case #{i} ({case['model']})**\n* **Extracted Text:** `{case['gt_text']}` == `{case['pred_text']}`\n* **Target Label:** `{case['gt_label']}`\n* **Predicted Label:** `{case['pred_label']}`\n---")
        elif q_num > 1:
            print("\n#### C. Label Mismatch")
            print("*No label mapping errors found for this question.*")

if __name__ == "__main__":
    run_error_analysis()
