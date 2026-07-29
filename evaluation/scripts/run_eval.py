import sys
import json
import re
import time
import pandas as pd
import warnings
import numpy as np
from collections import defaultdict
from pathlib import Path

from scripts.core import EvaluationEngine 
# from scripts.error_analysis import run_error_analysis
from scripts.visualization import (
    generate_performance_quadrant, 
    generate_verbosity_scatter,
    generate_task_heatmap_overall,
    generate_task_heatmap_hasans,
    generate_strict_vs_relaxed_quadrant,
    generate_classification_dropoff
)

warnings.filterwarnings("ignore")

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / 'data'
RAW_DIR = DATA_DIR / 'raw_inputs'
RESULTS_DIR = DATA_DIR / 'evaluation_results'

def discover_llm_files(raw_dir):
    llm_files_by_question = defaultdict(dict)
    for filepath in raw_dir.glob("*_results*.json"):
        match = re.search(r"^(.*?)_results(\d*)\.json$", filepath.name, re.IGNORECASE)
        if match:
            raw_model_name = match.group(1).replace("_", " ").replace("-", " ").title() 
            q_num = int(match.group(2)) if match.group(2) else 1 
            llm_files_by_question[q_num][raw_model_name] = filepath
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
    
    bert_queue_std, bert_queue_dd = [], []
    std_cross_pairs, dd_cross_pairs = [], []
    
    for model_name, filepath in model_files.items():
        if not filepath.exists(): continue
        
        try:
            with filepath.open('r', encoding='utf-8') as f: 
                predictions = json.load(f)
        except (json.JSONDecodeError, ValueError) as e:
            print(f"\n[CRITICAL WARNING] Skipping malformed file {filepath}: {str(e)}", file=sys.stderr)
            continue
            
        for pred_entry in predictions:
            article_text = pred_entry.get('source_article', '').strip()
            if article_text not in gt_map: continue
            
            gold_spans = gt_map[article_text].get('answer_labels', [])
            pred_spans = pred_entry.get('answer_labels', pred_entry.get('model_spans', [])) 

            g_texts = [g.get('text', '').strip() for g in gold_spans if g.get('text', '').strip()]
            p_texts = [p.get('text', '').strip() for p in pred_spans if p.get('text', '').strip()]
            
            has_ans = len(g_texts) > 0
            has_pred = len(p_texts) > 0
            
            rec_id = len(master_records)
            record = {
                "model": model_name, "question": q_num, "article": article_text,
                "has_ans": has_ans, "has_pred": has_pred,
                "g_texts": g_texts, "p_texts": p_texts,
                "g_labels": [str(g.get('labels', [])) for g in gold_spans],
                "p_labels": [str(p.get('labels', [])) for p in pred_spans],
                "spans_generated": len(p_texts),
                "em": 0.0, "iou": 0.0, "f1": 0.0, "dedup_f1": 0.0, "label_f1": 0.0
            }

            # SQuAD 2.0 Logic Routing
            if not has_ans and not has_pred:
                record["em"], record["iou"], record["f1"], record["dedup_f1"], record["label_f1"] = 1.0, 1.0, 1.0, 1.0, 1.0
            elif not has_ans and has_pred:
                record["em"], record["iou"], record["f1"], record["dedup_f1"], record["label_f1"] = 0.0, 0.0, 0.0, 0.0, 0.0
            elif has_ans and not has_pred:
                record["em"], record["iou"], record["f1"], record["dedup_f1"], record["label_f1"] = 0.0, 0.0, 0.0, 0.0, 0.0
            else:
                # Set-EM and Set-F1 via official SQuAD token normalization & math
                set_em, set_f1 = EvaluationEngine.evaluate_ie_squad_metrics(g_texts, p_texts)
                record["em"] = set_em
                record["iou"] = set_f1

                # Macro Label F1
                record["label_f1"] = EvaluationEngine.evaluate_strict_tuple_match(gold_spans, pred_spans)

                # Queue Pairs for BERTScore
                norm_p = [EvaluationEngine.normalize_answer(p) for p in p_texts]
                norm_g = [EvaluationEngine.normalize_answer(g) for g in g_texts]
                start_std = len(std_cross_pairs)
                std_cross_pairs.extend(zip([p for p in norm_p for g in norm_g], [g for p in norm_p for g in norm_g]))
                bert_queue_std.append((rec_id, start_std, len(norm_p), len(norm_g)))

                g_dedup = EvaluationEngine.deduplicate_texts(g_texts)
                p_dedup = EvaluationEngine.deduplicate_texts(p_texts)
                norm_p_d = [EvaluationEngine.normalize_answer(p) for p in p_dedup]
                norm_g_d = [EvaluationEngine.normalize_answer(g) for g in g_dedup]
                start_dd = len(dd_cross_pairs)
                dd_cross_pairs.extend(zip([p for p in norm_p_d for g in norm_g_d], [g for p in norm_p_d for g in norm_g_d]))
                bert_queue_dd.append((rec_id, start_dd, len(norm_p_d), len(norm_g_d)))

            master_records.append(record)

    print(f"    -> Pushing batched matrix to GPU...", file=sys.stderr, flush=True)
    global_f1_std = EvaluationEngine.run_global_bertscore_backend([x[0] for x in std_cross_pairs], [x[1] for x in std_cross_pairs]) if std_cross_pairs else []
    global_f1_dd = EvaluationEngine.run_global_bertscore_backend([x[0] for x in dd_cross_pairs], [x[1] for x in dd_cross_pairs]) if dd_cross_pairs else []

    for rec_id, start, num_p, num_g in bert_queue_std:
        if num_p == 0 or num_g == 0:
            master_records[rec_id]["f1"] = 0.0
            continue

        flat = global_f1_std[start : start + (num_p * num_g)]
        mat = np.array([flat[r * num_g : (r + 1) * num_g] for r in range(num_p)])
        mp, mr = mat.max(axis=1).mean(), mat.max(axis=0).mean()
        
        # Clip negative rescaled values BEFORE harmonic mean calculation
        mp = float(np.clip(mp, 0.0, 1.0))
        mr = float(np.clip(mr, 0.0, 1.0))
        master_records[rec_id]["f1"] = float(np.clip(round(2 * (mp * mr) / (mp + mr) if (mp + mr) > 0 else 0.0, 4), 0.0, 1.0))

    for rec_id, start, num_p, num_g in bert_queue_dd:
        if num_p == 0 or num_g == 0:
            master_records[rec_id]["dedup_f1"] = 0.0
            continue
            
        flat = global_f1_dd[start : start + (num_p * num_g)]
        mat = np.array([flat[r * num_g : (r + 1) * num_g] for r in range(num_p)])
        mp, mr = mat.max(axis=1).mean(), mat.max(axis=0).mean()
        
        # Clip negative rescaled values BEFORE harmonic mean calculation
        mp = float(np.clip(mp, 0.0, 1.0))
        mr = float(np.clip(mr, 0.0, 1.0))
        master_records[rec_id]["dedup_f1"] = float(np.clip(round(2 * (mp * mr) / (mp + mr) if (mp + mr) > 0 else 0.0, 4), 0.0, 1.0))
            
    return master_records


def aggregate_squad2_metrics(df, group_cols):
    """Aggregates metrics and returns both numeric columns (for plots) and formatted string columns (for tables)."""
    if df.empty:
        return pd.DataFrame()

    agg_data = []
    for name, group in df.groupby(group_cols):
        has_ans_group = group[group['has_ans'] == True]
        no_ans_group = group[group['has_ans'] == False]

        has_ans_n = len(has_ans_group)
        no_ans_n = len(no_ans_group)

        # 1. Abstention
        no_ans_acc = round(no_ans_group['em'].mean(), 4) if no_ans_n > 0 else 0.0

        # 2. SQuAD EM
        overall_em = round(group['em'].mean(), 4)
        has_ans_em = round(has_ans_group['em'].mean(), 4) if has_ans_n > 0 else 0.0

        # 3. Macro Label EM 
        overall_label_f1 = round(group['label_f1'].mean(), 4)
        has_ans_label_f1 = round(has_ans_group['label_f1'].mean(), 4) if has_ans_n > 0 else 0.0

        # 4. Token F1 / IoU
        overall_iou = round(group['iou'].mean(), 4)
        has_ans_iou = round(has_ans_group['iou'].mean(), 4) if has_ans_n > 0 else 0.0

        # 5. Std BERT
        overall_std_bert = round(group['f1'].mean(), 4)
        has_ans_std_bert = round(has_ans_group['f1'].mean(), 4) if has_ans_n > 0 else 0.0

        # 6. Dedup BERT
        overall_dd_bert = round(group['dedup_f1'].mean(), 4)
        has_ans_dd_bert = round(has_ans_group['dedup_f1'].mean(), 4) if has_ans_n > 0 else 0.0
        
        avg_spans = round(group['spans_generated'].mean(), 2)

        res = {
            "Model System": name if isinstance(name, str) else name[0],
            "Total_N": len(group),
            
            # Numeric columns used specifically for generating plots
            "Avg Spans": avg_spans,
            "NoAns Acc": no_ans_acc,
            "Overall Dedup BERT": overall_dd_bert,
            "HasAns Dedup BERT": has_ans_dd_bert, 
            "HasAns SQuAD EM": has_ans_em,
            "HasAns Token F1": has_ans_iou,
            "HasAns Std BERT": has_ans_std_bert,
            "HasAns Label F1": has_ans_label_f1,
            
            # Formatted columns used for the final Markdown table printout
            "Abstention (NoAns)": f"{no_ans_acc:.2f}",
            "SQuAD EM (Overall / HasAns)": f"{overall_em:.2f} / {has_ans_em:.2f}",
            "Label EM (Overall / HasAns)": f"{overall_label_f1:.2f} / {has_ans_label_f1:.2f}",
            "SQuAD Token F1 (Overall / HasAns)": f"{overall_iou:.2f} / {has_ans_iou:.2f}",
            "Standard BERTScore (Overall / HasAns)": f"{overall_std_bert:.2f} / {has_ans_std_bert:.2f}",
            "Deduped BERTScore (Overall / HasAns)": f"{overall_dd_bert:.2f} / {has_ans_dd_bert:.2f}"
        }
        
        if isinstance(name, tuple) and len(name) > 1:
            res["Question"] = name[1]

        agg_data.append(res)

    return pd.DataFrame(agg_data)


def run_evaluation_pipeline(question_filter=None):
    pipeline_start_time = time.time()
    llm_files = discover_llm_files(RAW_DIR)
    
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (BASE_DIR / "assets").mkdir(parents=True, exist_ok=True)
    
    all_records = []
    
    for q_num, model_files in sorted(llm_files.items()):
        if question_filter is not None and q_num != question_filter: continue
        print(f"\n--- Evaluating Question {q_num} ---", file=sys.stderr)
        
        gt_paths = [
            RAW_DIR / f'groundtruth_q{q_num}.json', 
            RAW_DIR / f'ground_truth_q{q_num}.json', 
            RAW_DIR / 'groundtruth.json'
        ]
        
        found_path = next((p for p in gt_paths if p.exists()), None)
        if not found_path:
            print(f"  [ERROR] No ground truth file found for Q{q_num}.", file=sys.stderr)
            continue
            
        gt_map = load_ground_truth_map(found_path)
        if not gt_map: continue
        
        all_records.extend(evaluate_models_globally(model_files, gt_map, q_num))

    artifact_path = RESULTS_DIR / "master_evaluation_artifact.json"
    with artifact_path.open("w", encoding="utf-8") as f:
        json.dump(all_records, f, indent=4)
        
    df = pd.DataFrame(all_records)
    if df.empty: 
        print(f"\n[FATAL] Pipeline failed to parse any records.", file=sys.stderr)
        return

    # Define exact columns to display 
    display_cols = [
        "Model System", 
        "Abstention (NoAns)", 
        "SQuAD EM (Overall / HasAns)", 
        "Label EM (Overall / HasAns)", 
        "SQuAD Token F1 (Overall / HasAns)",
        "Standard BERTScore (Overall / HasAns)",
        "Deduped BERTScore (Overall / HasAns)"
    ]

    # LAYER 1: GLOBAL AGGREGATION
    g_agg = aggregate_squad2_metrics(df, group_cols=['model'])

    # Pass the DF with numeric columns to the visualizer
    generate_performance_quadrant(df=g_agg, model_col="Model System", filename="master_quadrant.png")
    generate_strict_vs_relaxed_quadrant(df=g_agg, model_col="Model System", filename="strict_vs_relaxed.png")
    generate_verbosity_scatter(df=g_agg, model_col="Model System", filename="verbosity_vs_accuracy.png")

    total_time = round(time.time() - pipeline_start_time, 2)
    print(f"**Pipeline Execution Time:** {total_time} seconds")
    print(f"**Total Models Evaluated:** {len(g_agg)}")
    print(f"**Total Documents Processed:** {g_agg['Total_N'].max()}\n")

    print("## Part 1: Global Benchmark Leaderboard (SQuAD 2.0 Standard)")
    print("> *All text metrics formatted as (Overall / HasAns)*\n")
    # Safe sorting using the background numeric column
    print(g_agg.sort_values(by="Overall Dedup BERT", ascending=False)[display_cols].to_markdown(index=False))
    
    print("\n## Part 2: Visual Insights")
    print("\n### 1. Abstention vs. Extraction Quality")
    print("> *Evaluates whether models are 'Ideal Performers' (safe and accurate) or 'Hallucinators' (talkative but unsafe).*")
    print("![Master Performance Quadrant](assets/master_quadrant.png)\n")
    
    print("### 2. Strict vs. Relaxed Evaluation Shift")
    print("> *Visualizing the performance penalty models take when evaluated strictly (Exact Match) vs. relaxed (Token/Semantic).*")
    print("![Strict vs Relaxed](assets/strict_vs_relaxed.png)\n")
    
    print("### 3. Verbosity vs. Semantic Accuracy")
    print("> *Tracking whether models artificially inflate their extraction scores by over-generating spans.*")
    print("![Verbosity vs Semantic Accuracy](assets/verbosity_vs_accuracy.png)\n")

    # LAYER 2: PER-QUESTION AGGREGATION
    q_agg = aggregate_squad2_metrics(df, group_cols=['model', 'question'])
    unique_qs = sorted(q_agg['Question'].unique()) if 'Question' in q_agg.columns else []
    
    if len(unique_qs) > 1:
        print("\n---\n")
        print("## Part 3: Task Complexity Breakdown")
        
        heatmap_df = q_agg[['Model System', 'Question', 'Overall Dedup BERT', 'HasAns Dedup BERT']].copy()
        heatmap_df.rename(columns={'Model System': 'Model Target', 'Question': 'question'}, inplace=True)
        heatmap_df['Question Track'] = heatmap_df['question'].apply(lambda x: f"Question {x}")
        
        generate_task_heatmap_overall(df=heatmap_df, filename="task_complexity_heatmap_overall.png")
        generate_task_heatmap_hasans(df=heatmap_df, filename="task_complexity_heatmap_hasans.png")
        
        print("\n### Performance Degradation Heatmaps")
        print("> *Overall Score (includes easy abstentions) vs. HasAns Score (true extraction capability).*")
        print("![Task Complexity (Overall)](assets/task_complexity_heatmap_overall.png)")
        print("![Task Complexity (HasAns)](assets/task_complexity_heatmap_hasans.png)\n")

        # Generate the Drop-off chart (it handles the Q2 filtering internally)
        generate_classification_dropoff(df=q_agg, filename="classification_dropoff_q2.png")
        
        print("\n### Question 2: The Classification Penalty")
        print("> *Visualizing the gap between a model's ability to find the correct text vs. its ability to map it to the correct category.*")
        print("![Classification Dropoff](assets/classification_dropoff_q2.png)\n")

        for q in unique_qs:
            print(f"\n### Question {q} Leaderboard")
            q_subset = q_agg[q_agg['Question'] == q].copy()
            print(q_subset.sort_values(by="Overall Dedup BERT", ascending=False)[display_cols].to_markdown(index=False))
            print("\n")

    print("\n---\n")
    # run_error_analysis(records=all_records)

if __name__ == "__main__":
    run_evaluation_pipeline()