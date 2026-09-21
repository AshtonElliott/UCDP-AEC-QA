import sys
import json
import re
import time
import warnings
import numpy as np
from collections import defaultdict
from pathlib import Path
from scipy.optimize import linear_sum_assignment


from scripts.core import EvaluationEngine 

warnings.filterwarnings("ignore")

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / 'data'
RESULTS_DIR = DATA_DIR / 'evaluation_results'

def discover_llm_files(data_dir):
    llm_files_by_question = defaultdict(dict)
    
    subdirs = ["raw_NT", "raw_T", "cookbook_NT", "cookbook_T"]
    for subdir in subdirs:
        dir_path = data_dir / subdir
        if not dir_path.exists(): continue
        
        for filepath in dir_path.glob("*.json"):
            match = re.search(r"^(.*?)_results(\d*)(_cb)?(_NT|_T)?\.json$", filepath.name, re.IGNORECASE)
            if match:
                raw_model_name = match.group(1).replace("_", " ").replace("-", " ").title() 
                q_num = int(match.group(2)) if match.group(2) else 1 
                
                strategy = "cookbook" if "cookbook" in subdir else "raw"
                thinking = "T" if subdir.endswith("_T") else "NT"
                
                llm_files_by_question[q_num][(raw_model_name, strategy, thinking)] = filepath
                
    return llm_files_by_question

def load_ground_truth_map(filepath):
    with filepath.open('r', encoding='utf-8') as f: 
        gt_list = json.load(f)
    gt_map = {}
    for entry in gt_list:
        article_text = entry.get('source_article', entry.get('data', {}).get('source_article', '')).strip()
        if not article_text: continue
        gold_spans = entry.get('answer_labels', [])
        if not gold_spans and 'annotations' in entry:
            for anno in entry['annotations']:
                for res in anno.get('result', []):
                    text = res.get('value', {}).get('text', '').strip()
                    if text:
                        gold_spans.append({'text': text, 'labels': res['value'].get('labels', [])})
        gt_map[article_text] = {'answer_labels': gold_spans}
    return gt_map

def evaluate_models_globally(model_files, gt_map, q_num):
    master_records = []
    
    bert_queue_std = []
    std_cross_pairs = []
    
    for (model_name, strategy, thinking), filepath in model_files.items():
        if not filepath.exists(): continue
        
        try:
            with filepath.open('r', encoding='utf-8') as f: 
                predictions = json.load(f)
        except (json.JSONDecodeError, ValueError) as e:
            print(f"\n[CRITICAL WARNING] Skipping malformed file {str(filepath)}: {str(e)}", file=sys.stderr)
            continue
            
        for pred_entry in predictions:
            article_text = pred_entry.get('source_article', '').strip()
            
            # Rigorous match check - Skip extra inferences that don't belong in the consensus GT set
            if article_text not in gt_map: continue
            
            gold_spans = gt_map[article_text].get('answer_labels', [])
            pred_spans = pred_entry.get('answer_labels', pred_entry.get('model_spans', [])) 

            def _get_text_str(span):
                t = span.get('text', '')
                if isinstance(t, list):
                    return " ".join(str(x) for x in t) if t else ""
                return str(t)

            g_texts = [_get_text_str(g).strip() for g in gold_spans if _get_text_str(g).strip()]
            p_texts = [_get_text_str(p).strip() for p in pred_spans if _get_text_str(p).strip()]
            # Measure extracted spans before adding any scoring-only sentinel.
            spans_generated = len(p_texts)
            
            # Restore Hallucination Penalty: If the pipeline flagged a Non-Genuine (hallucinated) output
            no_ans_flag = pred_entry.get("no_answer", "")
            if "Non-Geniune" in no_ans_flag or "Non-Genuine" in no_ans_flag:
                p_texts = ["<HALLUCINATED_TEXT>"]
            
            has_ans = len(g_texts) > 0
            has_pred = len(p_texts) > 0
            
            rec_id = len(master_records)
            record = {
                "model": model_name, "strategy": strategy, "thinking": thinking, "question": q_num, "article": article_text,
                "has_ans": has_ans, "has_pred": has_pred,
                "g_texts": g_texts, "p_texts": p_texts,
                "g_labels": [str(g.get('labels', [])) for g in gold_spans],
                "p_labels": [str(p.get('labels', [])) for p in pred_spans],
                "spans_generated": spans_generated,
                "set_text_p": 0.0, "set_text_r": 0.0, "set_text_f1": 0.0, 
                "token_f1": 0.0, "bertscore_f1": 0.0, 
                "label_p": 0.0, "label_r": 0.0, "label_f1": 0.0,
                "cat_stats": {}
            }

            # Q2 Category Match Logic (Strict Tuple Extraction for Grid)
            g_tups = set()
            for g in gold_spans:
                t = EvaluationEngine.normalize_answer(_get_text_str(g))
                if not t: continue
                ls = g.get('labels', [])
                if isinstance(ls, list):
                    for l in ls: g_tups.add((t, str(l).strip()))
                elif isinstance(ls, str): g_tups.add((t, ls.strip()))
                else: g_tups.add((t, "no_label"))
                    
            p_tups = set()
            for p in pred_spans:
                t = EvaluationEngine.normalize_answer(_get_text_str(p))
                if not t: continue
                ls = p.get('labels', [])
                if isinstance(ls, list):
                    for l in ls: p_tups.add((t, str(l).strip()))
                elif isinstance(ls, str): p_tups.add((t, ls.strip()))
                else: p_tups.add((t, "no_label"))
                    
            cat_stats = defaultdict(lambda: {"TP": 0, "FP": 0, "FN": 0})
            for t, l in g_tups:
                if (t, l) in p_tups: cat_stats[l]["TP"] += 1
                else: cat_stats[l]["FN"] += 1
            for t, l in p_tups:
                if (t, l) not in g_tups: cat_stats[l]["FP"] += 1
            
            record["cat_stats"] = dict(cat_stats)

            # SQuAD 2.0 Logic Routing
            if not has_ans and not has_pred:
                record["set_text_p"], record["set_text_r"], record["set_text_f1"] = 1.0, 1.0, 1.0
                record["label_p"], record["label_r"], record["label_f1"] = 1.0, 1.0, 1.0
                record["token_f1"], record["bertscore_f1"] = 1.0, 1.0
            elif not has_ans and has_pred:
                record["set_text_p"], record["set_text_r"], record["set_text_f1"] = 0.0, 0.0, 0.0
                record["label_p"], record["label_r"], record["label_f1"] = 0.0, 0.0, 0.0
                record["token_f1"], record["bertscore_f1"] = 0.0, 0.0
            elif (has_ans and not has_pred) or (p_texts == ["<HALLUCINATED_TEXT>"]):
                record["set_text_p"], record["set_text_r"], record["set_text_f1"] = 0.0, 0.0, 0.0
                record["label_p"], record["label_r"], record["label_f1"] = 0.0, 0.0, 0.0
                record["token_f1"], record["bertscore_f1"] = 0.0, 0.0
            else:
                set_p, set_r, set_f1, tok_f1 = EvaluationEngine.evaluate_span_and_token_f1(g_texts, p_texts)
                record["set_text_p"] = set_p
                record["set_text_r"] = set_r
                record["set_text_f1"] = set_f1
                record["token_f1"] = tok_f1

                # Labeled Span metrics (text + category tuples)
                lbl_p, lbl_r, lbl_f1 = EvaluationEngine.evaluate_label_f1(gold_spans, pred_spans)
                record["label_p"] = lbl_p
                record["label_r"] = lbl_r
                record["label_f1"] = lbl_f1

                # Queue Pairs for BERTScore
                norm_p = [EvaluationEngine.normalize_answer(p) for p in p_texts]
                norm_g = [EvaluationEngine.normalize_answer(g) for g in g_texts]
                
                norm_p = [p for p in norm_p if p]
                norm_g = [g for g in norm_g if g]
                
                start_std = len(std_cross_pairs)
                std_cross_pairs.extend(zip([p for p in norm_p for g in norm_g], [g for p in norm_p for g in norm_g]))
                bert_queue_std.append((rec_id, start_std, len(norm_p), len(norm_g)))

            master_records.append(record)

    print(f"    -> Pushing batched matrix to GPU...", file=sys.stderr, flush=True)
    global_f1_std = EvaluationEngine.run_global_bertscore_backend([x[0] for x in std_cross_pairs], [x[1] for x in std_cross_pairs]) if std_cross_pairs else []
    
    # Process BERTScore
    for rec_id, start, num_p, num_g in bert_queue_std:
        if num_p == 0 and num_g == 0:
            master_records[rec_id]["bertscore_f1"] = 1.0
            continue
        if num_p == 0 or num_g == 0:
            master_records[rec_id]["bertscore_f1"] = 0.0
            continue

        flat = global_f1_std[start : start + (num_p * num_g)]
        mat = np.array([flat[r * num_g : (r + 1) * num_g] for r in range(num_p)])
        
        row_ind, col_ind = linear_sum_assignment(mat, maximize=True)
        matched_scores = mat[row_ind, col_ind]
        soft_tp = matched_scores[matched_scores > 0].sum()
        
        mp = float(np.clip(soft_tp / num_p if num_p > 0 else 0.0, 0.0, 1.0))
        mr = float(np.clip(soft_tp / num_g if num_g > 0 else 0.0, 0.0, 1.0))
        master_records[rec_id]["bertscore_f1"] = float(np.clip(round(2 * (mp * mr) / (mp + mr) if (mp + mr) > 0 else 0.0, 4), 0.0, 1.0))
            
    return master_records

def run_evaluation_pipeline(question_filter=None):
    pipeline_start_time = time.time()
    llm_files = discover_llm_files(DATA_DIR)
    
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (BASE_DIR / "assets").mkdir(parents=True, exist_ok=True)
    
    all_records = []
    
    for q_num, model_files in sorted(llm_files.items()):
        if question_filter is not None and q_num != question_filter: continue
        print(f"\n--- Computing Metrics for Question {q_num} ---", file=sys.stderr)
        
        gt_paths = [
            DATA_DIR / 'groundtruth' / f'ground_truth_q{q_num}.json', 
            DATA_DIR / 'groundtruth' / f'groundtruth_q{q_num}.json', 
            DATA_DIR / 'groundtruth' / 'groundtruth.json'
        ]
        
        found_path = next((p for p in gt_paths if p.exists()), None)
        if not found_path:
            print(f"  [ERROR] No ground truth file found for Q{q_num}.", file=sys.stderr)
            continue
            
        gt_map = load_ground_truth_map(found_path)
        if not gt_map: continue
        
        all_records.extend(evaluate_models_globally(model_files, gt_map, q_num))
    
    if not all_records:
        print("\n[FATAL] Pipeline failed to parse any records. Check your data paths.", file=sys.stderr)
        return

    artifact_path = RESULTS_DIR / "master_evaluation_artifact.json"
    with artifact_path.open("w", encoding="utf-8") as f:
        json.dump(all_records, f, indent=4)
        
    total_time = round(time.time() - pipeline_start_time, 2)
    
    # Save the pipeline execution time to a meta file so the report generator can pull it
    meta_path = RESULTS_DIR / "pipeline_metadata.json"
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump({"compute_time_seconds": total_time}, f, indent=4)
        
    print(f"[Compute Complete] GPU Pipeline Execution Time: {total_time} seconds", file=sys.stderr)

if __name__ == "__main__":
    run_evaluation_pipeline()