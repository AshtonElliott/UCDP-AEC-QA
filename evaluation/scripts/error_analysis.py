import sys
import json
from pathlib import Path
from collections import defaultdict

BASE_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = BASE_DIR / 'data' / 'evaluation_results'
ARTIFACT_PATH = RESULTS_DIR / "master_evaluation_artifact.json"

def run_error_analysis(records=None):
    print("\n## Part 4: Error Analysis & SQuAD 2.0 Edge Cases", file=sys.stdout)
    print("> *Automated extraction of specific failure modes across the IE pipeline.*", file=sys.stdout)
    
    if records is None:
        if not ARTIFACT_PATH.exists():
            return
        with ARTIFACT_PATH.open('r', encoding='utf-8') as f:
            records = json.load(f)
            
    # Structure: diagnostics[question_id]["error_type"] = [list of cases]
    diagnostics = defaultdict(lambda: {
        "paraphrase": [], 
        "hallucination": [], 
        "missed_extraction": [],
        "label_crash": []
    })

    for rec in records:
        q_num = rec.get('question', 1)
        g_text_str = " | ".join(rec['g_texts']) if rec['g_texts'] else "[NO TARGET SPANS EXIST]"
        p_text_str = " | ".join(rec['p_texts']) if rec['p_texts'] else "[NO SPANS PREDICTED]"
        
        # SQuAD 2.0 Boolean Flags
        has_ans = rec.get('has_ans', len(rec['g_texts']) > 0)
        has_pred = rec.get('has_pred', len(rec['p_texts']) > 0)

        # 1. SQuAD 2.0 Missed Extraction (Ground Truth Exists, but Model Abstained)
        if has_ans and not has_pred and len(diagnostics[q_num]["missed_extraction"]) < 3:
            diagnostics[q_num]["missed_extraction"].append({
                "model": rec['model'], "gt": g_text_str
            })

        # 2. SQuAD 2.0 Hallucination (Ground Truth is Empty, but Model Predicted Spans)
        elif not has_ans and has_pred and len(diagnostics[q_num]["hallucination"]) < 3:
            diagnostics[q_num]["hallucination"].append({
                "model": rec['model'], "pred": p_text_str, 
                "spans": rec['spans_generated']
            })

        # 3. Valid Paraphrasing (HasAns=True, High Semantic score, Zero Exact Match)
        # Using Set-EM == 0.0 and Deduped BERTScore > 0.70
        elif has_ans and has_pred and rec['em'] == 0.0 and rec['dedup_f1'] > 0.70 and len(diagnostics[q_num]["paraphrase"]) < 3:
            diagnostics[q_num]["paraphrase"].append({
                "model": rec['model'], "gt": g_text_str, "pred": p_text_str, 
                "em": rec['em'], "f1": rec['dedup_f1']
            })

        # 4. Label Mismatch (High text extraction match, but mapped to the wrong category)
        elif has_ans and has_pred:
            # strict_f1 represents the exact (normalized text + label) tuple match
            tp, fp, fn = rec.get('tp', 0), rec.get('fp', 0), rec.get('fn', 0)
            strict_tuple_f1 = (2 * tp) / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0.0
            
            # If Set-EM is high (text matches well) but tuple match is low (labels are wrong)
            if rec['em'] > 0.8 and strict_tuple_f1 < 0.3 and len(diagnostics[q_num]["label_crash"]) < 3:
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
        
        if q_data["missed_extraction"]:
            print("\n#### A. Missed Extractions (Failed to extract an existing answer)")
            for i, case in enumerate(q_data["missed_extraction"], 1):
                print(f"**Edge Case #{i} ({case['model']})**\n* **Ground Truth:** `{case['gt']}`\n* **Model Prediction:** `[ABSTAINED - Predicted Nothing]`\n---")

        if q_data["hallucination"]:
            print("\n#### B. Hallucinations (Generated text on an unanswerable article)")
            for i, case in enumerate(q_data["hallucination"], 1):
                print(f"**Edge Case #{i} ({case['model']})**\n* **Ground Truth:** `[EMPTY ARTICLE - Should have abstained]`\n* **Model Prediction:** `{case['pred']}`\n* **Spans Generated:** {case['spans']}\n---")
                
        if q_data["paraphrase"]:
            print("\n#### C. Valid Paraphrasing (High Semantic Match, Zero Exact Match)")
            for i, case in enumerate(q_data["paraphrase"], 1):
                print(f"**Edge Case #{i} ({case['model']})**\n* **Ground Truth:** `{case['gt']}`\n* **Model Prediction:** `{case['pred']}`\n* **Scores:** Set-EM = {case['em']:.4f} | Deduped BERTScore = {case['f1']:.4f}\n---")

        if q_data["label_crash"]:
            print("\n#### D. Label Mismatch (High Text Match, Wrong Category)")
            for i, case in enumerate(q_data["label_crash"], 1):
                print(f"**Edge Case #{i} ({case['model']})**\n* **Extracted Text:** `{case['gt_text']}` == `{case['pred_text']}`\n* **Target Label:** `{case['gt_label']}`\n* **Predicted Label:** `{case['pred_label']}`\n---")
        elif q_num > 1:
            print("\n#### D. Label Mismatch")
            print("*No label mapping errors found for this question.*")

if __name__ == "__main__":
    run_error_analysis()