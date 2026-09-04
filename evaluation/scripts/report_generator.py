import sys
import json
import time
import pandas as pd
from pathlib import Path

from scripts.visualization import (
    clear_assets_dir,
    # generate_performance_quadrant, 
    # generate_verbosity_scatter,
    # generate_task_heatmap_overall,
    # generate_task_heatmap_hasans,
    # generate_strict_vs_relaxed_quadrant,
    # generate_classification_dropoff,
    # generate_pr_bars,
    # generate_category_pr_grid
)
# from scripts.error_analysis import run_error_analysis

BASE_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = BASE_DIR / 'data' / 'evaluation_results'

def aggregate_squad2_metrics(df, group_cols):
    """Aggregates metrics including Precision and Recall for robust diagnostic analysis."""
    if df.empty:
        return pd.DataFrame()

    agg_data = []
    for name, group in df.groupby(group_cols):
        has_ans_group = group[group['has_ans'] == True]
        no_ans_group = group[group['has_ans'] == False]

        has_ans_n = len(has_ans_group)
        no_ans_n = len(no_ans_group)

        # 1. Abstention & Missed Rate
        no_ans_acc = round(no_ans_group['set_text_f1'].mean(), 4) if no_ans_n > 0 else 0.0
        missed_answer_rate = round((~has_ans_group['has_pred']).mean(), 4) if has_ans_n > 0 else 0.0

        # 2. Span F1, Precision, Recall
        overall_set_text_f1 = round(group['set_text_f1'].mean(), 4)
        has_ans_set_text_f1 = round(has_ans_group['set_text_f1'].mean(), 4) if has_ans_n > 0 else 0.0
        has_ans_set_text_p = round(has_ans_group['set_text_p'].mean(), 4) if has_ans_n > 0 else 0.0
        has_ans_set_text_r = round(has_ans_group['set_text_r'].mean(), 4) if has_ans_n > 0 else 0.0

        # 3. Labeled Span F1, Precision, Recall
        overall_label_f1 = round(group['label_f1'].mean(), 4)
        has_ans_label_f1 = round(has_ans_group['label_f1'].mean(), 4) if has_ans_n > 0 else 0.0
        has_ans_label_p = round(has_ans_group['label_p'].mean(), 4) if has_ans_n > 0 else 0.0
        has_ans_label_r = round(has_ans_group['label_r'].mean(), 4) if has_ans_n > 0 else 0.0

        # 4. Token F1
        overall_token_f1 = round(group['token_f1'].mean(), 4)
        has_ans_token_f1 = round(has_ans_group['token_f1'].mean(), 4) if has_ans_n > 0 else 0.0

        # 5. Std BERT
        overall_std_bert = round(group['bertscore_f1'].mean(), 4)
        has_ans_std_bert = round(has_ans_group['bertscore_f1'].mean(), 4) if has_ans_n > 0 else 0.0

        model_raw = name[group_cols.index('model')] if isinstance(name, tuple) and 'model' in group_cols else (name[0] if isinstance(name, tuple) else name)
        strategy_raw = name[group_cols.index('strategy')] if isinstance(name, tuple) and 'strategy' in group_cols else 'raw'
        thinking_raw = name[group_cols.index('thinking')] if isinstance(name, tuple) and 'thinking' in group_cols else 'NT'

        display_strategy = f"{strategy_raw.title()} ({thinking_raw})"

        res = {
            "Model": model_raw,
            "Method": display_strategy,
            "Model_Raw": model_raw,          
            "Strategy_Raw": strategy_raw,
            "Thinking_Raw": thinking_raw,
            "Doc Count": len(group),
            "NoAns Acc": no_ans_acc,
            "Missed Answer Rate Num": missed_answer_rate,
            "HasAns Span F1": has_ans_set_text_f1,
            "HasAns Span P": has_ans_set_text_p,
            "HasAns Span R": has_ans_set_text_r,
            "HasAns Token F1": has_ans_token_f1,
            "HasAns BERTScore": has_ans_std_bert,
            "HasAns Labeled Span F1": has_ans_label_f1,
            "HasAns Labeled P": has_ans_label_p,
            "HasAns Labeled R": has_ans_label_r,
            
            "Abstention (NoAns)": f"{no_ans_acc:.2f}",
            "Missed Answer Rate": f"{missed_answer_rate:.2f}",
            "Span F1 (Overall / HasAns)": f"{overall_set_text_f1:.2f} / {has_ans_set_text_f1:.2f}",
            "Labeled Span F1 (Overall / HasAns)": f"{overall_label_f1:.2f} / {has_ans_label_f1:.2f}",
            "SQuAD Token F1 (Overall / HasAns)": f"{overall_token_f1:.2f} / {has_ans_token_f1:.2f}",
            "BERTScore (Overall / HasAns)": f"{overall_std_bert:.2f} / {has_ans_std_bert:.2f}"
        }
        
        if isinstance(name, tuple) and 'question' in group_cols:
            res["Question"] = int(name[group_cols.index('question')])

        agg_data.append(res)

    return pd.DataFrame(agg_data)

def sort_paired_leaderboard(df):
    """Rank by raw HasAns BERTScore; nest each model's cookbook row directly underneath.

    When a Question column is present, the raw anchor is per (model, question)
    so Q1/Q2 tables are not polluted by scores from the other question.
    Cookbook-only models fall back to their own HasAns BERTScore.
    """
    if df.empty:
        return df

    df = df.copy()
    zero_shot = df[df['Strategy_Raw'] == 'raw']
    has_question = 'Question' in df.columns

    if has_question:
        std_perfs = zero_shot.set_index(['Model_Raw', 'Question'])['HasAns BERTScore'].to_dict()
        df['Sort_Key'] = df.apply(
            lambda r: std_perfs.get((r['Model_Raw'], r['Question']), r['HasAns BERTScore']),
            axis=1,
        )
        return df.sort_values(
            by=['Question', 'Sort_Key', 'Model_Raw', 'Strategy_Raw', 'Thinking_Raw'],
            ascending=[True, False, True, False, False],
        )

    std_perfs = zero_shot.set_index('Model_Raw')['HasAns BERTScore'].to_dict()
    df['Sort_Key'] = df.apply(
        lambda r: std_perfs.get(r['Model_Raw'], r['HasAns BERTScore']),
        axis=1,
    )
    # Strategy_Raw descending puts raw before cookbook (r > c)
    return df.sort_values(
        by=['Sort_Key', 'Model_Raw', 'Strategy_Raw', 'Thinking_Raw'],
        ascending=[False, True, False, False],
    )

def format_grouped_table_for_display(df, display_cols):
    """
    Safely blanks out repeating Model names for consecutive paired method rows,
    while guaranteeing standalone models (no cookbook) never get blanked out.
    """
    table_df = df[display_cols].copy()
    
    formatted_models = []
    previous_model = None
    
    for current_model in df['Model_Raw']:
        if current_model != previous_model:
            formatted_models.append(current_model)
            previous_model = current_model
        else:
            formatted_models.append("")
            
    table_df['Model'] = formatted_models
    return table_df

def generate_full_report():
    report_start_time = time.time()
    artifact_path = RESULTS_DIR / "master_evaluation_artifact.json"
    
    if not artifact_path.exists():
        print("[ERROR] No artifact found. Run the compute pipeline first.", file=sys.stderr)
        return

    # Clear old graphics out of the assets directory
    clear_assets_dir()

    with artifact_path.open("r", encoding="utf-8") as f:
        all_records = json.load(f)
        
    df = pd.DataFrame(all_records)
    if df.empty: 
        print("\n[FATAL] Artifact is empty.", file=sys.stderr)
        return

    # Extract the GPU compute time logged by report_generator.py
    meta_path = RESULTS_DIR / "pipeline_metadata.json"
    compute_time = "Unknown"
    if meta_path.exists():
        with meta_path.open("r", encoding="utf-8") as f:
            meta = json.load(f)
            compute_time = meta.get("compute_time_seconds", "Unknown")

    # ENSURE TYPE CONSISTENCY FOR QUESTION NUMBERS & STRATEGY
    if 'question' in df.columns:
        df['question'] = pd.to_numeric(df['question'], errors='coerce').fillna(0).astype(int)
        
    # Convert 'standard' or 'zero-shot' strings in old datasets directly to 'raw'.
    if 'strategy' not in df.columns:
        df['strategy'] = 'raw'
    else:
        df['strategy'] = df['strategy'].replace({'standard': 'raw', 'zero-shot': 'raw'})
        
    if 'thinking' not in df.columns:
        df['thinking'] = 'NT'

    display_cols = [
        "Model", "Method", "Doc Count", "Abstention (NoAns)", "Missed Answer Rate",
        "Span F1 (Overall / HasAns)", "Labeled Span F1 (Overall / HasAns)", 
        "SQuAD Token F1 (Overall / HasAns)", "BERTScore (Overall / HasAns)"
    ]

    # LAYER 1: GLOBAL AGGREGATION
    g_agg = aggregate_squad2_metrics(df, group_cols=['model', 'strategy', 'thinking'])
    g_agg = sort_paired_leaderboard(g_agg)

    print(f"**Total Models Evaluated:** {df['model'].nunique()}")
    print(f"**Total Documents Processed:** {g_agg['Doc Count'].max()}")
    print(f"**Full Evaluation Compute Time:** {compute_time} seconds\n")

    # Table 1: Global Benchmark Leaderboard
    print("## Part 1: Global Benchmark Leaderboard (SQuAD 2.0 Evaluation)")
    print("> *All text metrics formatted as (Overall / HasAns)*\n")
    print(format_grouped_table_for_display(g_agg, display_cols).to_markdown(index=False))

    # LAYER 2: PER-QUESTION AGGREGATION
    q_agg = aggregate_squad2_metrics(df, group_cols=['model', 'strategy', 'thinking', 'question'])

    if 'Question' in q_agg.columns:
        unique_qs = sorted(q_agg['Question'].unique())
        
        if len(unique_qs) > 0:
            print("\n---\n")
            print("## Part 2: Task Complexity Breakdown")

            for q in unique_qs:
                q_num = int(q)
                print(f"\n### Question {q_num} Leaderboard")
                q_subset = sort_paired_leaderboard(q_agg[q_agg['Question'] == q_num].copy())
                print(format_grouped_table_for_display(q_subset, display_cols).to_markdown(index=False))

    print("\n---\n")
    
    total_time = round(time.time() - report_start_time, 2)
    print(f"\n[Report System] Report rendering completed in {total_time} seconds", file=sys.stderr)

if __name__ == "__main__":
    generate_full_report()