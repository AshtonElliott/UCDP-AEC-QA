import os
import sys
import json
import glob
import re
import time
import pandas as pd
import warnings
from collections import defaultdict

from scripts.core import EvaluationEngine 
from scripts.visualization import (
    generate_leaderboard_bar, 
    generate_performance_quadrant, 
    generate_scaling_plot
)

warnings.filterwarnings("ignore")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) if '__file__' in locals() else os.getcwd()
DATA_DIR = os.path.join(BASE_DIR, 'data')
RAW_DIR = os.path.join(DATA_DIR, 'raw_inputs')

def discover_llm_files(raw_dir):
    llm_files_by_question = defaultdict(dict)
    pattern = os.path.join(raw_dir, "*_results*.json")
    for filepath in glob.glob(pattern):
        filename = os.path.basename(filepath)
        match = re.search(r"^(.*?)_results(\d*)\.json$", filename, re.IGNORECASE)
        if match:
            raw_model_name = match.group(1).replace("_", " ").title() 
            q_num_str = match.group(2)
            q_num = int(q_num_str) if q_num_str else 1 
            llm_files_by_question[q_num][raw_model_name] = filepath
    return llm_files_by_question

def load_ground_truth_map(filepath):
    if not os.path.exists(filepath): return {}
    with open(filepath, 'r', encoding='utf-8') as f: 
        gt_list = json.load(f)
        
    gt_map = {}
    for entry in gt_list:
        article_text = entry.get('source_article', '')
        if not article_text and 'data' in entry:
            article_text = entry['data'].get('source_article', '')
            
        article_text = article_text.strip()
        if not article_text: continue

        gold_spans = []
        if 'answer_labels' in entry:
            gold_spans = entry['answer_labels']
        elif 'annotations' in entry:
            for anno in entry['annotations']:
                for res in anno.get('result', []):
                    val = res.get('value', {})
                    text = val.get('text', '').strip()
                    if text:
                        gold_spans.append({'text': text, 'labels': val.get('labels', [])})
                        
        gt_map[article_text] = {'answer_labels': gold_spans}
    return gt_map

def evaluate_single_model(model_name, filepath, gt_map):
    if not os.path.exists(filepath): return None, []
    with open(filepath, 'r', encoding='utf-8') as f: 
        llm_predictions = json.load(f)

    metrics = {
        "squad_em_scores": [], "f1_scores": [], 
        "iou_scores": [], "f1_dedup_scores": [],
        "tp": 0, "fp": 0, "fn": 0
    }
    
    text_pairs = []

    for pred_entry in llm_predictions:
        article_text = pred_entry.get('source_article', '').strip()
        if article_text not in gt_map: continue 

        gt_entry = gt_map[article_text]
        gold_spans = gt_entry.get('answer_labels', [])
        pred_spans = pred_entry.get('answer_labels', pred_entry.get('model_spans', [])) 

        tp, fp, fn = EvaluationEngine.evaluate_strict_entity_match(gold_spans, pred_spans)
        metrics["tp"] += tp
        metrics["fp"] += fp
        metrics["fn"] += fn

        g_texts = [g.get('text', '').strip() for g in gold_spans if g.get('text', '').strip()]
        p_texts = [p.get('text', '').strip() for p in pred_spans if p.get('text', '').strip()]
        
        text_pairs.append((g_texts, p_texts))

        metrics["squad_em_scores"].append(EvaluationEngine.evaluate_exact_match(g_texts, p_texts))
        _, _, f1 = EvaluationEngine.evaluate_bipartite_bertscore(g_texts, p_texts)
        metrics["f1_scores"].append(f1)
        metrics["iou_scores"].append(EvaluationEngine.evaluate_iou_match(g_texts, p_texts))
        _, _, f1_dedup = EvaluationEngine.evaluate_dedup_bertscore(g_texts, p_texts)
        metrics["f1_dedup_scores"].append(f1_dedup)

    return metrics, text_pairs

def run_pipeline_stress_test(pools_by_question):
    """Benchmarks each question tracking group independently to identify explicit complexity variances."""
    print("Executing distinct computational stress tests per question...", file=sys.stderr)
    
    data_by_question = {}
    profile_rows = []
    
    for q_num, pool in sorted(pools_by_question.items()):
        total_records = len(pool)
        if total_records < 5: continue
        
        batch_sizes = [int(total_records * (i / 5)) for i in range(1, 6)]
        recorded_times = []
        
        for size in batch_sizes:
            sliced_pool = pool[:size]
            start_clock = time.time()
            
            for g_texts, p_texts in sliced_pool:
                _ = EvaluationEngine.evaluate_exact_match(g_texts, p_texts)
                _ = EvaluationEngine.evaluate_dedup_bertscore(g_texts, p_texts)
                
            recorded_times.append(time.time() - start_clock)
            
        data_by_question[q_num] = {"sizes": batch_sizes, "times": recorded_times}
        
        for n, t in zip(batch_sizes, recorded_times):
            profile_rows.append({
                "Evaluation Track": f"Question {q_num}",
                "Samples (N)": n,
                "Compute Time (s)": round(t, 2),
                "Throughput": f"{round(n / t, 2)} it/s" if t > 0 else "0 it/s"
            })
            
    if not data_by_question:
        return None
        
    generate_scaling_plot(data_by_question, filename="pipeline_scaling.png")
    return pd.DataFrame(profile_rows)

def run_comparative_pipeline(question_filter=None):
    print("Scanning for LLM inference files...", file=sys.stderr)
    pipeline_start_time = time.time()
    
    llm_files_by_question = discover_llm_files(RAW_DIR)
    if not llm_files_by_question:
        print("No *_results*.json files found.", file=sys.stderr)
        return
        
    os.makedirs("assets", exist_ok=True)
    
    aggregate_tracker = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0, "em_sum": 0, "f1_sum": 0, "iou_sum": 0, "dedup_sum": 0, "count": 0, "time_sum": 0.0})
    pools_by_question = {}

    for q_num, model_files in sorted(llm_files_by_question.items()):
        if question_filter is not None and q_num != question_filter: continue
            
        print(f"\n--- Evaluating Question {q_num} ---", file=sys.stderr)
        
        gt_paths = [
            os.path.join(RAW_DIR, f'train_q{q_num}.json'), os.path.join(RAW_DIR, f'ground_truth_q{q_num}.json'),
            os.path.join(RAW_DIR, 'train.json'), os.path.join(RAW_DIR, 'ground_truth.json')
        ]
        
        gt_map = next((load_ground_truth_map(p) for p in gt_paths if os.path.exists(p)), None)
        if not gt_map:
            print(f"Skipping Q{q_num}: No GT file.", file=sys.stderr)
            continue
            
        q_pool_captured = False
        q_results = []
        
        for model_name, file_path in model_files.items():
            print(f"  Processing {model_name}...", end="\r", file=sys.stderr)
            
            model_start_time = time.time()
            m, text_pairs = evaluate_single_model(model_name, file_path, gt_map)
            model_end_time = time.time()
            
            if not m or not m["squad_em_scores"]: continue
            
            if not q_pool_captured and text_pairs:
                pools_by_question[q_num] = text_pairs
                q_pool_captured = True
            
            elapsed_sec = model_end_time - model_start_time
            total_eval = len(m["squad_em_scores"])
            throughput = total_eval / elapsed_sec if elapsed_sec > 0 else 0
            
            precision = m["tp"] / (m["tp"] + m["fp"]) if (m["tp"] + m["fp"]) > 0 else 0.0
            recall = m["tp"] / (m["tp"] + m["fn"]) if (m["tp"] + m["fn"]) > 0 else 0.0
            strict_f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
            
            q_results.append({
                "Model Target": f"**{model_name}**", 
                "Samples": total_eval, 
                "Strict Labeled F1": round(strict_f1, 4), 
                "Lexical (EM)": round(sum(m["squad_em_scores"]) / total_eval, 4), 
                "Token IoU": round(sum(m["iou_scores"]) / total_eval, 4),
                "Semantic (F1)": round(sum(m["f1_scores"]) / total_eval, 4),
                "Dedup F1": round(sum(m["f1_dedup_scores"]) / total_eval, 4),
                "Time (s)": round(elapsed_sec, 2),
                "Throughput": f"{round(throughput, 1)} it/s"
            })
            
            agg = aggregate_tracker[model_name]
            agg["tp"] += m["tp"]; agg["fp"] += m["fp"]; agg["fn"] += m["fn"]
            agg["em_sum"] += sum(m["squad_em_scores"])
            agg["f1_sum"] += sum(m["f1_scores"])
            agg["iou_sum"] += sum(m["iou_scores"])
            agg["dedup_sum"] += sum(m["f1_dedup_scores"])
            agg["count"] += total_eval
            agg["time_sum"] += elapsed_sec
                
        if q_results:
            print(f"  Model processing complete for Q{q_num}!        ", file=sys.stderr)
            df_q = pd.DataFrame(q_results).sort_values(by="Dedup F1", ascending=False)
            print(f"\n#### Breakdown Matrix: Question Track {q_num}")
            print(df_q.to_markdown(index=False))

    # ==========================================
    # GENERATE THE MASTER AGGREGATE RESULTS
    # ==========================================
    if aggregate_tracker:
        print("\n--- Generating Master Benchmark Overview ---", file=sys.stderr)
        master_results = []
        for model, a in aggregate_tracker.items():
            if a["count"] == 0: continue
            prec = a["tp"] / (a["tp"] + a["fp"]) if (a["tp"] + a["fp"]) > 0 else 0.0
            rec = a["tp"] / (a["tp"] + a["fn"]) if (a["tp"] + a["fn"]) > 0 else 0.0
            overall_f1 = 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
            
            master_results.append({
                "Model System": f"**{model}**", 
                "Total N": a["count"], 
                "Composite Labeled F1": round(overall_f1, 4),
                "Lexical (EM)": round(a["em_sum"] / a["count"], 4),
                "Token IoU": round(a["iou_sum"] / a["count"], 4),
                "Contextual F1": round(a["f1_sum"] / a["count"], 4),
                "Deduped F1": round(a["dedup_sum"] / a["count"], 4),
                "Compute (s)": round(a["time_sum"], 2),
                "Avg Throughput": f"{round(a["count"] / a["time_sum"], 1)} it/s" if a["time_sum"] > 0 else "0 it/s"
            })
            
        df_master = pd.DataFrame(master_results).sort_values(by="Deduped F1", ascending=False)
        
        generate_leaderboard_bar(df=df_master, model_col="Model System", filename="master_leaderboard.png")
        generate_performance_quadrant(df=df_master, model_col="Model System", filename="master_quadrant.png")

        print(f"\n### Consolidated Master Leaderboard (Global Aggregate Mapping)\n")
        print(df_master.to_markdown(index=False))
        
        pipeline_elapsed = round(time.time() - pipeline_start_time, 2)
        print(f"\n*Total Pipeline Execution Duration: {pipeline_elapsed} seconds*\n")
        
        print(f"\n### Evaluation Capture Projections\n")
        print(f"![Master Leaderboard](assets/master_leaderboard.png)\n")
        print(f"![Master Performance Quadrant](assets/master_quadrant.png)\n")

        # ==========================================
        # AUTOMATED COMPREHENSIVE MULTI-LINE STRESS TEST
        # ==========================================
        if pools_by_question:
            print("\n## 4. Computational Scaling Profile & Stress Test\n")
            df_stress = run_pipeline_stress_test(pools_by_question)
            if df_stress is not None:
                print(df_stress.to_markdown(index=False))
                print("\n![Linear Cost Model](assets/pipeline_scaling.png)\n")

if __name__ == "__main__":
    run_comparative_pipeline()