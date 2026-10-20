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
    plot_reasoning_vs_prompting_comparison,
    plot_bert_pretraining_impact,
    plot_bert_casing_impact,
    plot_head_to_head_showdown
)

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

        if strategy_raw == 'extractive' or any(b in str(model_raw).lower() for b in ['bert', 'confli']):
            display_strategy = "Extractive QA"
        else:
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
    Assign Cookbook-only and Extractive models their own HasAns BERTScore.
    """
    if df.empty:
        return df

    df = df.copy()
    raw_df = df[df['Strategy_Raw'] == 'raw']
    has_question = 'Question' in df.columns

    if has_question:
        std_perfs = raw_df.set_index(['Model_Raw', 'Question'])['HasAns BERTScore'].to_dict()
        df['Sort_Key'] = df.apply(
            lambda r: std_perfs.get((r['Model_Raw'], r['Question']), r['HasAns BERTScore']),
            axis=1,
        )
        return df.sort_values(
            by=['Question', 'Sort_Key', 'Model_Raw', 'Strategy_Raw', 'Thinking_Raw'],
            ascending=[True, False, True, False, False],
        )

    std_perfs = raw_df.set_index('Model_Raw')['HasAns BERTScore'].to_dict()
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

def generate_full_report(mode="all"):
    report_start_time = time.time()
    artifact_path = RESULTS_DIR / "master_evaluation_artifact.json"
    
    if not artifact_path.exists():
        print("[ERROR] No artifact found. Run the compute pipeline first.", file=sys.stderr)
        return

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

    # Ensure model_type column exists
    if 'model_type' not in df.columns:
        df['model_type'] = df['model'].apply(lambda m: 'bert' if any(b in str(m).lower() for b in ['bert', 'confli']) else 'generative')

    # Enforce type consistency for question numbers and strategy.
    if 'question' in df.columns:
        df['question'] = pd.to_numeric(df['question'], errors='coerce').fillna(0).astype(int)
        
    if 'strategy' not in df.columns:
        df['strategy'] = 'raw'
    else:
        df['strategy'] = df['strategy'].replace({'standard': 'raw', 'zero-shot': 'raw'})
        
    if 'thinking' not in df.columns:
        df['thinking'] = 'NT'

    # Filter by mode
    if mode == "bert":
        df = df[df['model_type'] == 'bert'].copy()
        report_title = "# ConfliBERT & BERT Evaluation Report"
    elif mode == "generative":
        df = df[df['model_type'] == 'generative'].copy()
        report_title = "# LLM Generative Evaluation Report"
    else:
        report_title = "# Master Evaluation Report: Generative LLMs vs. Domain-Adapted Encoders"

    if df.empty:
        print(f"[FATAL] No records found for mode '{mode}'. Try running the pipeline for this model type.", file=sys.stderr)
        return

    display_cols = [
        "Model", "Method", "Doc Count", "Abstention (NoAns)", "Missed Answer Rate",
        "Span F1 (Overall / HasAns)", "Labeled Span F1 (Overall / HasAns)", 
        "SQuAD Token F1 (Overall / HasAns)", "BERTScore (Overall / HasAns)"
    ]

    # Global aggregation
    g_agg = aggregate_squad2_metrics(df, group_cols=['model', 'strategy', 'thinking'])
    g_agg = sort_paired_leaderboard(g_agg)

    print(report_title)
    print(f"**Total Models Evaluated:** {df['model'].nunique()}")
    print(f"**Total Documents Processed:** {g_agg['Doc Count'].max() if not g_agg.empty else 0}")
    print(f"**Full Evaluation Compute Time:** {compute_time} seconds\n")

    # Part 1 Table
    print("## Part 1: Global Benchmark Leaderboard (SQuAD 2.0 Evaluation)")
    print("> *All text metrics formatted as (Overall / HasAns)*\n")
    print(format_grouped_table_for_display(g_agg, display_cols).to_markdown(index=False))

    # Part 2 Table (Per-Question)
    q_agg = aggregate_squad2_metrics(df, group_cols=['model', 'strategy', 'thinking', 'question'])

    if 'Question' in q_agg.columns:
        unique_qs = sorted(q_agg['Question'].unique())
        
        if len(unique_qs) > 0:
            print("\n---\n")
            print("## Part 2: Task Complexity Breakdown")

            for q in unique_qs:
                q_num = int(q)
                title_suffix = "Weapon & Force Extraction" if q_num == 1 else "Infrastructure Damage & Categorization"
                print(f"\n### Question {q_num} Leaderboard ({title_suffix})")
                q_subset = sort_paired_leaderboard(q_agg[q_agg['Question'] == q_num].copy())
                print(format_grouped_table_for_display(q_subset, display_cols).to_markdown(index=False))

    print("\n---\n")
    print("## Part 3: Graphical Diagnostics\n")

    if mode == "all":
        # Section 3.1: Cross-Paradigm Comparisons
        print("### Section 3.1: Overall Comparisons (LLMs vs. Extractive Encoders)\n")
        
        print("Generating Figure 1: Overall Ranking: ConfliBERT vs. Generative LLMs...")
        print("> Compares the best setup of each model architecture (ConfliBERT, BERT-base, and generative LLMs), ranked by extraction quality (BERTScore F1).\n")
        plot_head_to_head_showdown(g_agg)
        print("![Figure 1: Overall Ranking: ConfliBERT vs. Generative LLMs](./assets/fig5_head_to_head_showdown.png)\n")

        print("Generating Figure 2: Model Reliability and Output Length...")
        print("> Left: Accuracy vs. knowing when to abstain on empty texts (avoiding hallucinations). Right: Does extracting more spans improve quality?\n")
        plot_hallucination_and_verbosity(g_agg, df, mode=mode)
        print("![Figure 2: Model Reliability and Output Length](./assets/fig2_hallucination_and_verbosity.png)\n")

        if 'Question' in q_agg.columns and 2 in q_agg['Question'].values:
            print("Generating Figure 3: Question 2: Finding Text vs. Labeling Categories...")
            print("> Shows how accuracy drops when models must not only extract the text span, but also assign its correct category.\n")
            plot_classification_accuracy_drop(q_agg, mode=mode)
            print("![Figure 3: Question 2: Finding Text vs. Labeling Categories](./assets/fig3_classification_accuracy_drop.png)\n")

        # Section 3.2: Generative LLM Analysis
        print("### Section 3.2: LLM Prompting & Reasoning\n")
        
        print("Generating Figure 4: Prompt Strategy Impact: Raw vs. Cookbook...")
        print("> Compares Raw against Cookbook prompts across generative models without thinking tokens.\n")
        plot_prompt_engineering_impact(g_agg)
        print("![Figure 4: Prompt Strategy Impact: Raw vs. Cookbook](./assets/fig1_prompt_engineering_impact.png)\n")

        if 'Thinking_Raw' in q_agg.columns and 'T' in q_agg['Thinking_Raw'].values:
            print("Generating Figure 5: Thinking Models vs. Prompt Strategy on Task Complexity...")
            print("> Compares whether reasoning tokens (Thinking mode) or prompt strategy (Raw vs. Cookbook) help more on Q1 (simpler) vs. Q2 (complex).\n")
            plot_reasoning_vs_prompting_comparison(q_agg)
            print("![Figure 5: Thinking Models vs. Prompt Strategy](./assets/fig4_reasoning_vs_prompting.png)\n")

        # Section 3.3: Extractive ConfliBERT & BERT Analysis
        print("### Section 3.3: ConfliBERT & BERT Diagnostics\n")
        
        print("Generating Figure 6: Pretraining Impact: General BERT vs. Conflict ConfliBERT...")
        print("> Compares general-domain BERT-base against conflict-pretrained ConfliBERT (Continual & Scratch) on Question 1 and Question 2.\n")
        plot_bert_pretraining_impact(q_agg)
        print("![Figure 6: Pretraining Impact: General BERT vs. Conflict ConfliBERT](./assets/fig_b1_pretraining_impact.png)\n")

        print("Generating Figure 7: Case Sensitivity: Cased vs. Uncased Models...")
        print("> Shows whether preserving letter case (acronyms and proper nouns) improves exact text match (Span F1) over lowercase-only models.\n")
        plot_bert_casing_impact(q_agg)
        print("![Figure 7: Case Sensitivity: Cased vs. Uncased Models](./assets/fig_b2_casing_impact.png)\n")

    elif mode == "generative":
        print("### Section 3.1: Reliability and Output Length\n")
        
        print("Generating Figure 1: Model Reliability and Output Length...")
        print("> Left: Accuracy vs. knowing when to abstain on empty texts (avoiding hallucinations). Right: Does extracting more spans improve quality?\n")
        plot_hallucination_and_verbosity(g_agg, df, mode=mode)
        print("![Figure 1: Model Reliability and Output Length](./assets/fig2_hallucination_and_verbosity.png)\n")

        if 'Question' in q_agg.columns and 2 in q_agg['Question'].values:
            print("Generating Figure 2: Question 2: Finding Text vs. Labeling Categories...")
            print("> Shows how accuracy drops when models must not only extract the text span, but also assign its correct category.\n")
            plot_classification_accuracy_drop(q_agg, mode=mode)
            print("![Figure 2: Question 2: Finding Text vs. Labeling Categories](./assets/fig3_classification_accuracy_drop.png)\n")

        print("### Section 3.2: LLM Prompting & Reasoning\n")

        print("Generating Figure 3: Prompt Strategy Impact: Raw vs. Cookbook...")
        print("> Compares Raw against Cookbook prompts across generative models without thinking tokens.\n")
        plot_prompt_engineering_impact(g_agg)
        print("![Figure 3: Prompt Strategy Impact: Raw vs. Cookbook](./assets/fig1_prompt_engineering_impact.png)\n")

        if 'Thinking_Raw' in q_agg.columns and 'T' in q_agg['Thinking_Raw'].values:
            print("Generating Figure 4: Thinking Models vs. Prompt Strategy on Task Complexity...")
            print("> Compares whether reasoning tokens (Thinking mode) or prompt strategy (Raw vs. Cookbook) help more on Q1 (simpler) vs. Q2 (complex).\n")
            plot_reasoning_vs_prompting_comparison(q_agg)
            print("![Figure 4: Thinking Models vs. Prompt Strategy](./assets/fig4_reasoning_vs_prompting.png)\n")

    elif mode == "bert":
        print("### Section 3.1: Reliability and Output Length\n")
        
        print("Generating Figure 1: Model Reliability and Output Length...")
        print("> Left: Accuracy vs. knowing when to abstain on empty texts (avoiding hallucinations). Right: Does extracting more spans improve quality?\n")
        plot_hallucination_and_verbosity(g_agg, df, mode=mode)
        print("![Figure 1: Model Reliability and Output Length](./assets/fig2_hallucination_and_verbosity.png)\n")

        if 'Question' in q_agg.columns and 2 in q_agg['Question'].values:
            print("Generating Figure 2: Question 2: Finding Text vs. Labeling Categories...")
            print("> Shows how accuracy drops when models must not only extract the text span, but also assign its correct category.\n")
            plot_classification_accuracy_drop(q_agg, mode=mode)
            print("![Figure 2: Question 2: Finding Text vs. Labeling Categories](./assets/fig3_classification_accuracy_drop.png)\n")

        print("### Section 3.2: ConfliBERT & BERT Diagnostics\n")

        print("Generating Figure 3: Pretraining Impact: General BERT vs. Conflict ConfliBERT...")
        print("> Compares general-domain BERT-base against conflict-pretrained ConfliBERT (Continual & Scratch) on Question 1 and Question 2.\n")
        plot_bert_pretraining_impact(q_agg)
        print("![Figure 3: Pretraining Impact: General BERT vs. Conflict ConfliBERT](./assets/fig_b1_pretraining_impact.png)\n")

        print("Generating Figure 4: Case Sensitivity: Cased vs. Uncased Models...")
        print("> Shows whether preserving letter case (acronyms and proper nouns) improves exact text match (Span F1) over lowercase-only models.\n")
        plot_bert_casing_impact(q_agg)
        print("![Figure 4: Case Sensitivity: Cased vs. Uncased Models](./assets/fig_b2_casing_impact.png)\n")

    total_time = round(time.time() - report_start_time, 2)
    print(f"\n[Report System] Report rendering completed in {total_time} seconds", file=sys.stderr)

if __name__ == "__main__":
    generate_full_report()