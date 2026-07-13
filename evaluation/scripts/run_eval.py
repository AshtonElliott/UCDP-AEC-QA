import os
import sys
import json
import glob
import re
import time
import pandas as pd
import warnings
import numpy as np
from collections import defaultdict
from pathlib import Path

from scripts.core import EvaluationEngine 
from scripts.error_analysis import run_error_analysis
from scripts.visualization import (
    generate_performance_quadrant, 
    generate_verbosity_scatter,
    generate_task_heatmap
)

warnings.filterwarnings("ignore")

# 🚀 FOOLPROOF PATHING
BASE_DIR = str(Path(__file__).resolve().parent.parent)
DATA_DIR = os.path.join(BASE_DIR, 'data')
RAW_DIR = os.path.join(DATA_DIR, 'raw_inputs')
RESULTS_DIR = os.path.join(DATA_DIR, 'evaluation_results')

def discover_llm_files(raw_dir):
    llm_files_by_question = defaultdict(dict)
    for filepath in glob.glob(os.path.join(raw_dir, "*_results*.json")):
        match = re.search(r"^(.*?)_results(\d*)\.json$", os.path.basename(filepath), re.IGNORECASE)
        if match:
            # 🚀 FIX: Aggressive normalization to prevent duplicate model profiles
            raw_model_name = match.group(1).replace("_", " ").replace("-", " ").title() 
            q_num = int(match.group(2)) if match.group(2) else 1 
            llm_files_by_question[q_num][raw_model_name] = filepath
    return llm_files_by_question

def load_ground_truth_map(filepath):
    with open(filepath, 'r', encoding='utf-8') as f: 
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
    standard_cross_pairs, dedup_cross_pairs = [], []
    standard_slices, dedup_slices = [], []
    
    for model_name, filepath in model_files.items():
        if not os.path.exists(filepath): continue
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f: 
                predictions = json.load(f)
        except (json.JSONDecodeError, ValueError) as e:
            print(f"\n[CRITICAL WARNING] Skipping malformed file {filepath}: {str(e)}", file=sys.stderr)
            continue
            
        for pred_entry in predictions:
            article_text = pred_entry.get('source_article', '').strip()
            if article_text not in gt_map: continue
            
            gold_spans = gt_map[article_text].get('answer_labels', [])
            pred_spans = pred_entry.get('answer_labels', pred_entry.get('model_spans', [])) 

            if not gold_spans and not pred_spans:
                strict_f1 = 1.0
                tp, fp, fn = 0, 0, 0
            else:
                tp, fp, fn = EvaluationEngine.evaluate_strict_entity_match(gold_spans, pred_spans)
                strict_prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                strict_rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                strict_f1 = 2 * (strict_prec * strict_rec) / (strict_prec + strict_rec) if (strict_prec + strict_rec) > 0 else 0.0

            g_texts = [g.get('text', '').strip() for g in gold_spans if g.get('text', '').strip()]
            p_texts = [p.get('text', '').strip() for p in pred_spans if p.get('text', '').strip()]
            
            squad_em = EvaluationEngine.evaluate_exact_match(g_texts, p_texts)
            iou = EvaluationEngine.evaluate_iou_match(g_texts, p_texts)

            norm_p = [EvaluationEngine.normalize_answer(p) for p in p_texts]
            norm_g = [EvaluationEngine.normalize_answer(g) for g in g_texts]
            standard_slices.append((len(master_records), len(standard_cross_pairs), len(norm_p), len(norm_g)))
            standard_cross_pairs.extend(zip([p for p in norm_p for g in norm_g], [g for p in norm_p for g in norm_g]))

            g_dedup = EvaluationEngine.deduplicate_texts(g_texts)
            p_dedup = EvaluationEngine.deduplicate_texts(p_texts)
            norm_p_d = [EvaluationEngine.normalize_answer(p) for p in p_dedup]
            norm_g_d = [EvaluationEngine.normalize_answer(g) for g in g_dedup]
            dedup_slices.append((len(master_records), len(dedup_cross_pairs), len(norm_p_d), len(norm_g_d)))
            dedup_cross_pairs.extend(zip([p for p in norm_p_d for g in norm_g_d], [g for p in norm_p_d for g in norm_g_d]))

            master_records.append({
                "model": model_name, "question": q_num, "article": article_text,
                "em": squad_em, "iou": iou, "strict_f1": strict_f1, "tp": tp, "fp": fp, "fn": fn,
                "g_texts": g_texts, "p_texts": p_texts,
                "g_labels": [str(g.get('labels', [])) for g in gold_spans],
                "p_labels": [str(p.get('labels', [])) for p in pred_spans],
                "spans_generated": len(p_texts)
            })

    print(f"    -> Pushing batched matrix to GPU...", file=sys.stderr, flush=True)
    global_f1_std = EvaluationEngine.run_global_bertscore_backend([x[0] for x in standard_cross_pairs], [x[1] for x in standard_cross_pairs]) if standard_cross_pairs else []
    global_f1_dedup = EvaluationEngine.run_global_bertscore_backend([x[0] for x in dedup_cross_pairs], [x[1] for x in dedup_cross_pairs]) if dedup_cross_pairs else []

    for idx, (rec_id, start_std, num_p, num_g) in enumerate(standard_slices):
        if num_p == 0 or num_g == 0:
            master_records[rec_id]["f1"] = 0.0
        else:
            flat = global_f1_std[start_std : start_std + (num_p * num_g)]
            mat = np.array([flat[r * num_g : (r + 1) * num_g] for r in range(num_p)])
            mp, mr = mat.max(axis=1).mean(), mat.max(axis=0).mean()
            master_records[rec_id]["f1"] = float(np.clip(round(2 * (mp * mr) / (mp + mr) if (mp + mr) > 0 else 0.0, 4), 0.0, 1.0))

    for idx, (rec_id, start_dd, num_p, num_g) in enumerate(dedup_slices):
        if num_p == 0 or num_g == 0:
            master_records[rec_id]["dedup_f1"] = 0.0
        else:
            flat = global_f1_dedup[start_dd : start_dd + (num_p * num_g)]
            mat = np.array([flat[r * num_g : (r + 1) * num_g] for r in range(num_p)])
            mp, mr = mat.max(axis=1).mean(), mat.max(axis=0).mean()
            master_records[rec_id]["dedup_f1"] = float(np.clip(round(2 * (mp * mr) / (mp + mr) if (mp + mr) > 0 else 0.0, 4), 0.0, 1.0))
            
    return master_records

def run_evaluation_pipeline(question_filter=None):
    pipeline_start_time = time.time()
    llm_files = discover_llm_files(RAW_DIR)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(os.path.join(BASE_DIR, "assets"), exist_ok=True)
    
    all_records = []
    
    for q_num, model_files in sorted(llm_files.items()):
        if question_filter is not None and q_num != question_filter: continue
        print(f"\n--- Evaluating Question {q_num} ---", file=sys.stderr)
        
        gt_paths = [
            os.path.join(RAW_DIR, f'groundtruth_q{q_num}.json'), 
            os.path.join(RAW_DIR, f'ground_truth_q{q_num}.json'), 
            os.path.join(RAW_DIR, 'groundtruth.json')
        ]
        
        found_path = next((p for p in gt_paths if os.path.exists(p)), None)
        if not found_path:
            print(f"  [ERROR] No ground truth file found for Q{q_num}.", file=sys.stderr)
            continue
            
        gt_map = load_ground_truth_map(found_path)
        if not gt_map: continue
        
        all_records.extend(evaluate_models_globally(model_files, gt_map, q_num))

    artifact_path = os.path.join(RESULTS_DIR, "master_evaluation_artifact.json")
    with open(artifact_path, "w", encoding="utf-8") as f:
        json.dump(all_records, f, indent=4)
        
    df = pd.DataFrame(all_records)
    if df.empty: 
        print(f"\n[FATAL] Pipeline failed to parse any records.", file=sys.stderr)
        return

    def safe_global_f1(row):
        tp, fp, fn = row['tp'], row['fp'], row['fn']
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        return round(2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0, 4)

    display_cols = ["Model System", "Total_N", "Label EM", "Lexical (EM)", "Token IoU", "Contextual F1", "Deduped F1", "Avg Spans"]

    # 🚀 LAYER 1: GLOBAL AGGREGATION (Restored)
    g_agg = df.groupby('model').agg(
        Total_N=('article', 'count'),
        em_sum=('em', 'sum'), iou_sum=('iou', 'sum'), f1_sum=('f1', 'sum'), dedup_sum=('dedup_f1', 'sum'),
        span_sum=('spans_generated', 'sum'), tp=('tp', 'sum'), fp=('fp', 'sum'), fn=('fn', 'sum')
    ).reset_index()

    g_agg['Label EM'] = g_agg.apply(safe_global_f1, axis=1)
    g_agg['Lexical (EM)'] = round(g_agg['em_sum'] / g_agg['Total_N'], 4)
    g_agg['Token IoU'] = round(g_agg['iou_sum'] / g_agg['Total_N'], 4)
    g_agg['Contextual F1'] = round(g_agg['f1_sum'] / g_agg['Total_N'], 4)
    g_agg['Deduped F1'] = round(g_agg['dedup_sum'] / g_agg['Total_N'], 4)
    g_agg['Avg Spans'] = round(g_agg['span_sum'] / g_agg['Total_N'], 2)
    g_agg.rename(columns={'model': 'Model System'}, inplace=True)

    generate_performance_quadrant(df=g_agg, model_col="Model System", filename="master_quadrant.png")
    generate_verbosity_scatter(df=g_agg, model_col="Model System", filename="verbosity_vs_accuracy.png")

    total_time = round(time.time() - pipeline_start_time, 2)

    # 🚀 THE NEW SMOOTH REPORT FLOW (Engineering Tone)
    print(f"**Pipeline Execution Time:** {total_time} seconds")
    print(f"**Total Models Evaluated:** {len(g_agg)}")
    print(f"**Total Documents Processed:** {g_agg['Total_N'].max()}\n")

    print("## Part 1: Global Benchmark Leaderboard")
    print("> *Overall system performance aggregated across all tasks.*\n")
    print(g_agg[display_cols].sort_values(by="Deduped F1", ascending=False).to_markdown(index=False))
    
    print("\n## Part 2: Visual Insights")
    print("\n### 1. Strict vs. Relaxed Evaluation Shift")
    print("> *Models shifting right demonstrate strong exact-word retrieval. Models shifting up demonstrate strong contextual understanding, even if phrasing differs from the ground truth.*")
    print("![Master Performance Quadrant](assets/master_quadrant.png)\n")
    
    print("### 2. Verbosity vs. Semantic Accuracy")
    print("> *Tracking whether models artificially inflate their semantic coverage by over-generating spans.*")
    print("![Verbosity vs Semantic Accuracy](assets/verbosity_vs_accuracy.png)\n")

    # 🚀 LAYER 2: PER-QUESTION AGGREGATION
    q_agg = df.groupby(['model', 'question']).agg(
        Total_N=('article', 'count'),
        em_sum=('em', 'sum'), iou_sum=('iou', 'sum'), f1_sum=('f1', 'sum'), dedup_sum=('dedup_f1', 'sum'),
        span_sum=('spans_generated', 'sum'), tp=('tp', 'sum'), fp=('fp', 'sum'), fn=('fn', 'sum')
    ).reset_index()

    q_agg['Label EM'] = q_agg.apply(safe_global_f1, axis=1)
    q_agg['Lexical (EM)'] = round(q_agg['em_sum'] / q_agg['Total_N'], 4)
    q_agg['Token IoU'] = round(q_agg['iou_sum'] / q_agg['Total_N'], 4)
    q_agg['Contextual F1'] = round(q_agg['f1_sum'] / q_agg['Total_N'], 4)
    q_agg['Deduped F1'] = round(q_agg['dedup_sum'] / q_agg['Total_N'], 4)
    q_agg['Avg Spans'] = round(q_agg['span_sum'] / q_agg['Total_N'], 2)

    unique_qs = sorted(q_agg['question'].unique())
    
    if len(unique_qs) > 1:
        print("\n---\n")
        print("## Part 3: Task Complexity Breakdown")
        
        heatmap_df = q_agg[['model', 'question', 'Deduped F1']].copy()
        heatmap_df.rename(columns={'model': 'Model Target', 'Deduped F1': 'Dedup F1'}, inplace=True)
        heatmap_df['Question Track'] = heatmap_df['question'].apply(lambda x: f"Question {x}")
        
        generate_task_heatmap(df=heatmap_df, filename="task_complexity_heatmap.png")
        print("\n### Performance Degradation Heatmap")
        print("> *Visualizing how well models maintain accuracy as the task shifts from simple extraction to complex classification.*")
        print("![Task Complexity Heatmap](assets/task_complexity_heatmap.png)\n")

        for q in unique_qs:
            print(f"\n### Question {q} Leaderboard")
            q_subset = q_agg[q_agg['question'] == q].copy()
            q_subset.rename(columns={'model': 'Model System'}, inplace=True)
            print(q_subset[display_cols].sort_values(by="Deduped F1", ascending=False).to_markdown(index=False))
            print("\n")

    print("\n---\n")
    # Part 4 prints directly from the error_analysis script
    run_error_analysis(records=all_records)

if __name__ == "__main__":
    run_evaluation_pipeline()