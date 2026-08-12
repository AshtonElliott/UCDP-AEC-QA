import sys
import json
import pandas as pd
from pathlib import Path

from scripts.visualization import (
    generate_performance_quadrant, 
    generate_verbosity_scatter,
    generate_task_heatmap_overall,
    generate_task_heatmap_hasans,
    generate_strict_vs_relaxed_quadrant,
    generate_classification_dropoff,
    generate_pr_bars,
    generate_category_pr_grid
)

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
        overall_std_bert = round(group['f1'].mean(), 4)
        has_ans_std_bert = round(has_ans_group['f1'].mean(), 4) if has_ans_n > 0 else 0.0

        # 6. Dedup BERT
        overall_dd_bert = round(group['dedup_f1'].mean(), 4)
        has_ans_dd_bert = round(has_ans_group['dedup_f1'].mean(), 4) if has_ans_n > 0 else 0.0
        
        avg_spans = round(group['spans_generated'].mean(), 2)

        res = {
            "Model": name if isinstance(name, str) else name[0],
            "Doc Count": len(group),
            "Avg Spans": avg_spans,
            "NoAns Acc": no_ans_acc,
            "Missed Answer Rate Num": missed_answer_rate,
            "Overall Dedup BERT": overall_dd_bert,
            "HasAns Dedup BERT": has_ans_dd_bert, 
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
            "BERTScore (Overall / HasAns)": f"{overall_std_bert:.2f} / {has_ans_std_bert:.2f}",
            "Deduped BERTScore (Overall / HasAns)": f"{overall_dd_bert:.2f} / {has_ans_dd_bert:.2f}"
        }
        
        if isinstance(name, tuple) and len(name) > 1:
            res["Question"] = int(name[1])

        agg_data.append(res)

    return pd.DataFrame(agg_data)


def generate_full_report():
    artifact_path = RESULTS_DIR / "master_evaluation_artifact.json"
    
    if not artifact_path.exists():
        print("[ERROR] No artifact found. Run the compute pipeline first.", file=sys.stderr)
        return

    with artifact_path.open("r", encoding="utf-8") as f:
        all_records = json.load(f)
        
    df = pd.DataFrame(all_records)
    if df.empty: 
        print("\n[FATAL] Artifact is empty.", file=sys.stderr)
        return

    # ENSURE TYPE CONSISTENCY FOR QUESTION NUMBERS
    if 'question' in df.columns:
        df['question'] = pd.to_numeric(df['question'], errors='coerce').fillna(0).astype(int)

    display_cols = [
        "Model", "Doc Count", "Avg Spans", "Abstention (NoAns)", "Missed Answer Rate",
        "Span F1 (Overall / HasAns)", "Labeled Span F1 (Overall / HasAns)", 
        "SQuAD Token F1 (Overall / HasAns)", "BERTScore (Overall / HasAns)",
        "Deduped BERTScore (Overall / HasAns)"
    ]

    # LAYER 1: GLOBAL AGGREGATION
    g_agg = aggregate_squad2_metrics(df, group_cols=['model'])

    # Generate Visuals
    generate_performance_quadrant(df=g_agg, model_col="Model", filename="master_quadrant.png")
    generate_strict_vs_relaxed_quadrant(df=g_agg, model_col="Model", filename="strict_vs_relaxed.png")
    generate_verbosity_scatter(df=g_agg, model_col="Model", filename="verbosity_vs_accuracy.png")

    print(f"**Total Models Evaluated:** {len(g_agg)}")
    print(f"**Total Documents Processed:** {g_agg['Doc Count'].max()}\n")

    print("## Part 1: Global Benchmark Leaderboard (SQuAD 2.0 Standard)")
    print("> *All text metrics formatted as (Overall / HasAns)*\n")
    print(g_agg.sort_values(by="HasAns Dedup BERT", ascending=False)[display_cols].to_markdown(index=False))
    
    print("\n## Part 2: Visual Insights")
    print("\n### 1. Abstention vs. Extraction Quality")
    print("> *Evaluates whether models are 'Ideal Performers' (safe and accurate) or 'Hallucinators' (talkative but unsafe).*")
    print("![Master Performance Quadrant](assets/master_quadrant.png)\n")
    
    print("### 2. Strict vs. Relaxed Evaluation Shift")
    print("> *Visualizing the performance penalty models take when evaluated strictly (Span F1) vs. relaxed (Token/Semantic).*")
    print("![Strict vs Relaxed](assets/strict_vs_relaxed.png)\n")
    
    print("### 3. Verbosity vs. Semantic Accuracy")
    print("> *Tracking whether models artificially inflate their extraction scores by over-generating spans.*")
    print("![Verbosity vs Semantic Accuracy](assets/verbosity_vs_accuracy.png)\n")

    # LAYER 2: PER-QUESTION AGGREGATION
    q_agg = aggregate_squad2_metrics(df, group_cols=['model', 'question'])
    if 'Question' in q_agg.columns:
        unique_qs = sorted(q_agg['Question'].unique())
        
        if len(unique_qs) > 0:
            print("\n---\n")
            print("## Part 3: Task Complexity Breakdown")
            
            heatmap_df = q_agg[['Model', 'Question', 'Overall Dedup BERT', 'HasAns Dedup BERT']].copy()
            heatmap_df.rename(columns={'Model': 'Model Target', 'Question': 'question'}, inplace=True)
            heatmap_df['Question Track'] = heatmap_df['question'].apply(lambda x: f"Question {x}")
            
            generate_task_heatmap_overall(df=heatmap_df, filename="task_complexity_heatmap_overall.png")
            generate_task_heatmap_hasans(df=heatmap_df, filename="task_complexity_heatmap_hasans.png")
            
            print("\n### Performance Degradation Heatmaps")
            print("> *Overall Score (includes easy abstentions) vs. HasAns Score (true extraction capability).*")
            print("![Task Complexity (Overall)](assets/task_complexity_heatmap_overall.png)")
            print("![Task Complexity (HasAns)](assets/task_complexity_heatmap_hasans.png)\n")

            for q in unique_qs:
                q_num = int(q)
                print(f"\n### Question {q_num} Leaderboard")
                q_subset = q_agg[q_agg['Question'] == q_num].copy()
                print(q_subset.sort_values(by="HasAns Dedup BERT", ascending=False)[display_cols].to_markdown(index=False))
                
                # Plot A: HasAns precision/recall bars (single operating point per model)
                bars_filename = f"q{q_num}_pr_bars.png"
                generate_pr_bars(
                    q_subset, 
                    "HasAns Span R", 
                    "HasAns Span P", 
                    f"Q{q_num}: Precision vs. Recall (Text Spans)",
                    bars_filename
                )
                print(f"\n#### Plot A: Text Extraction Precision & Recall (Q{q_num})")
                print("> *This isolates reading comprehension: Did the model locate the correct phrases? HasAns-only bars at exact span match — one operating point per model.*")
                print(f"![Plot A: Text Spans](assets/{bars_filename})\n")
                    
            # Generate Q2 Category Grid & Dropoff ONCE outside the per-question loop
            df_q2 = df[df['question'] == 2]
            if not df_q2.empty:
                generate_category_pr_grid(df_q2, "q2_category_pr_grid.png")
                print("#### Plot B: Category Mislabeling Breakdown")
                print("> *Macro-averages hide class-level failures. This 8-panel grid isolates which specific event labels models confuse after extracting the text.*")
                print("![Plot B: Category Grid](assets/q2_category_pr_grid.png)\n")

                generate_classification_dropoff(df=q_agg, filename="classification_dropoff_q2.png")
                print("\n### Question 2: The Classification Penalty")
                print("> **Classification Dropoff = set_text_f1 - label_f1**")
                print("> *A large gap indicates the model successfully acts as a search engine (finding the correct evidence text) but fails as a classifier (assigning the wrong event label).*")
                print("![Classification Dropoff](assets/classification_dropoff_q2.png)\n")

if __name__ == "__main__":
    generate_full_report()