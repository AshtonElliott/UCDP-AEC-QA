import sys
import json
import time
import pandas as pd
from pathlib import Path

from scripts.visualization import (
    clear_assets_dir,
    plot_prompt_engineering_impact,
    plot_hallucination_and_verbosity,
    plot_classification_accuracy_drop,
    plot_reasoning_vs_prompting_comparison
)
# from scripts.error_analysis import run_error_analysis

BASE_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = BASE_DIR / 'data' / 'evaluation_results'

def aggregate_squad2_metrics(df, group_cols):
    """Aggregate metrics including Precision and Recall for diagnostic analysis."""
    if df.empty:
        return pd.DataFrame()

    agg_data = []
    for name, group in df.groupby(group_cols):
        has_ans_group = group[group['has_ans'] == True]
        no_ans_group = group[group['has_ans'] == False]

        has_ans_n = len(has_ans_group)
        no_ans_n = len(no_ans_group)

        # Calculate Abstention and Missed Rate.
        no_ans_acc = round(no_ans_group['set_text_f1'].mean(), 4) if no_ans_n > 0 else 0.0
        missed_answer_rate = round((~has_ans_group['has_pred']).mean(), 4) if has_ans_n > 0 else 0.0

        # Calculate Span F1, Precision, and Recall.
        overall_set_text_f1 = round(group['set_text_f1'].mean(), 4)
        has_ans_set_text_f1 = round(has_ans_group['set_text_f1'].mean(), 4) if has_ans_n > 0 else 0.0
        has_ans_set_text_p = round(has_ans_group['set_text_p'].mean(), 4) if has_ans_n > 0 else 0.0
        has_ans_set_text_r = round(has_ans_group['set_text_r'].mean(), 4) if has_ans_n > 0 else 0.0

        # Calculate Labeled Span F1, Precision, and Recall.
        overall_label_f1 = round(group['label_f1'].mean(), 4)
        has_ans_label_f1 = round(has_ans_group['label_f1'].mean(), 4) if has_ans_n > 0 else 0.0
        has_ans_label_p = round(has_ans_group['label_p'].mean(), 4) if has_ans_n > 0 else 0.0
        has_ans_label_r = round(has_ans_group['label_r'].mean(), 4) if has_ans_n > 0 else 0.0

        # Calculate Token F1.
        overall_token_f1 = round(group['token_f1'].mean(), 4)
        has_ans_token_f1 = round(has_ans_group['token_f1'].mean(), 4) if has_ans_n > 0 else 0.0

        # Calculate Standard BERTScore.
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
    """Rank models by raw HasAns BERTScore and group cookbook variants.
    Determine raw anchor per (model, question) when a Question column is present.
    Assign Cookbook-only models their own HasAns BERTScore.
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
    # Sort Strategy_Raw in descending order to position raw before cookbook.
    return df.sort_values(
        by=['Sort_Key', 'Model_Raw', 'Strategy_Raw', 'Thinking_Raw'],
        ascending=[False, True, False, False],
    )

def format_grouped_table_for_display(df, display_cols):
    """
    Remove repeating Model names for consecutive paired method rows.
    Preserve standalone models that lack cookbook variants.
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

    # Clear previous graphics from the assets directory.
    clear_assets_dir()

    with artifact_path.open("r", encoding="utf-8") as f:
        all_records = json.load(f)
        
    df = pd.DataFrame(all_records)
    if df.empty: 
        print("\n[FATAL] Artifact is empty.", file=sys.stderr)
        return

    # Extract GPU compute time from pipeline metadata.
    meta_path = RESULTS_DIR / "pipeline_metadata.json"
    compute_time = "Unknown"
    if meta_path.exists():
        with meta_path.open("r", encoding="utf-8") as f:
            meta = json.load(f)
            compute_time = meta.get("compute_time_seconds", "Unknown")

    # Enforce type consistency for question numbers and strategy.
    if 'question' in df.columns:
        df['question'] = pd.to_numeric(df['question'], errors='coerce').fillna(0).astype(int)
        
    # Convert legacy 'standard' or 'zero-shot' identifiers to 'raw'.
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

    # Execute global aggregation.
    g_agg = aggregate_squad2_metrics(df, group_cols=['model', 'strategy', 'thinking'])
    g_agg = sort_paired_leaderboard(g_agg)

    print(f"**Total Models Evaluated:** {df['model'].nunique()}")
    print(f"**Total Documents Processed:** {g_agg['Doc Count'].max()}")
    print(f"**Full Evaluation Compute Time:** {compute_time} seconds\n")

    # Generate Global Benchmark Leaderboard table.
    print("## Part 1: Global Benchmark Leaderboard (SQuAD 2.0 Evaluation)")
    print("> *All text metrics formatted as (Overall / HasAns)*\n")
    print(format_grouped_table_for_display(g_agg, display_cols).to_markdown(index=False))

    # Execute per-question aggregation.
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
    print("## Part 3: Graphical Diagnostics\n")
    
    print("Generating Figure 1: Prompt Engineering Impact...")
    print("> This dumbbell plot shows if using prompt engineering (Cookbook) actually improves a model's extraction accuracy compared to its raw baseline.\n")
    plot_prompt_engineering_impact(g_agg)
    print("![Figure 1: Prompt Engineering Impact](./assets/fig1_prompt_engineering_impact.png)\n")

    print("Generating Figure 2: Hallucination and Verbosity...")
    print("> These two panels show whether models fail by hallucinating on empty texts, and how their verbosity directly links to those errors.\n")
    plot_hallucination_and_verbosity(g_agg, df)
    print("![Figure 2: Hallucination and Verbosity](./assets/fig2_hallucination_and_verbosity.png)\n")

    if 'Question' in q_agg.columns and 2 in q_agg['Question'].values:
        print("Generating Figure 3: Classification Accuracy Drop...")
        print("> This bar chart focuses on Question 2 to show the performance penalty when a model successfully extracts the correct text but assigns the wrong category.\n")
        plot_classification_accuracy_drop(q_agg)
        print("![Figure 3: Classification Accuracy Drop](./assets/fig3_classification_accuracy_drop.png)\n")

    if 'Thinking_Raw' in q_agg.columns and 'T' in q_agg['Thinking_Raw'].values:
        print("Generating Figure 4: Reasoning vs. Prompting Comparison...")
        print("> This faceted bar chart compares only the thinking-capable models to see whether adding reasoning tokens or prompt engineering provides the bigger boost.\n")
        plot_reasoning_vs_prompting_comparison(q_agg)
        print("![Figure 4: Reasoning vs. Prompting Comparison](./assets/fig4_reasoning_vs_prompting.png)\n")

    total_time = round(time.time() - report_start_time, 2)
    print(f"\n[Report System] Report rendering completed in {total_time} seconds", file=sys.stderr)

if __name__ == "__main__":
    generate_full_report()