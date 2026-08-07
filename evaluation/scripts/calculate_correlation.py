import os
import sys
import glob
import pandas as pd
import warnings
from scipy.stats import spearmanr

from scripts.core import EvaluationEngine
from scripts.visualization import clear_assets_dir, generate_performance_quadrant

os.environ["TOKENIZERS_PARALLELISM"] = "false" 
warnings.filterwarnings("ignore")

SCRIPT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) if '__file__' in locals() else os.getcwd()
DATA_DIR = os.path.join(SCRIPT_DIR, 'data')
COMPLETED_DIR = os.path.join(DATA_DIR, 'completed_grading')
MASTER_TEMPLATE = os.path.join(DATA_DIR, "grading_templates", "master_grading_template.csv")
KEY_FILE = os.path.join(DATA_DIR, "evaluation_results", "secret_decryption_key.csv")

def get_evaluation_data():
    print("loading human rater sheets...", file=sys.stderr)
    csv_files = glob.glob(os.path.join(COMPLETED_DIR, "*.csv"))
    if not csv_files:
        return None

    all_scores = []
    rater_names = []
    for file in csv_files:
        df_temp = pd.read_csv(file)
        if 'Human_Score_1_to_5' in df_temp.columns:
            rater_name = os.path.basename(file).split('_')[0]
            rater_names.append(rater_name)
            clean = df_temp[['Evaluation_ID', 'Human_Score_1_to_5']].dropna()
            all_scores.append(clean)
    
    if not all_scores:
        return None

    df_all_raters = pd.concat(all_scores)
    df_consolidated = df_all_raters.groupby('Evaluation_ID', as_index=False).agg(
        Consolidated_Human_Score=('Human_Score_1_to_5', 'mean')
    )
    
    if not os.path.exists(MASTER_TEMPLATE) or not os.path.exists(KEY_FILE):
        return None
        
    df_template = pd.read_csv(MASTER_TEMPLATE)
    df_key = pd.read_csv(KEY_FILE)

    df_master = pd.merge(df_template, df_consolidated, on="Evaluation_ID", how="inner")
    df_master = pd.merge(df_master, df_key, on="Evaluation_ID", how="inner")
    df_master["Normalized_Human_Score"] = (df_master["Consolidated_Human_Score"] - 1) / 4

    f1_scores = []
    set_text_f1_scores = []
    token_f1_scores = []
    f1_dedup_scores = []
    
    for idx, row in df_master.iterrows():
        print(f"Processing article {idx + 1}/{len(df_master)}...", end="\r", file=sys.stderr)
        
        human_str = str(row.get("Human_Ground_Truth", "")).strip()
        llm_str = str(row.get("LLM_Prediction", "")).strip()

        g_texts = [] if human_str in ["", "nan", "NO ANSWER"] else [s.strip() for s in human_str.split(" | ") if s.strip()]
        p_texts = [] if llm_str in ["", "nan", "NO ANSWER"] else [s.strip() for s in llm_str.split(" | ") if s.strip()]

        set_text_f1, token_f1 = EvaluationEngine.evaluate_ie_squad_metrics(g_texts, p_texts)
        set_text_f1_scores.append(set_text_f1)
        _, _, f1 = EvaluationEngine.evaluate_bipartite_bertscore(g_texts, p_texts)
        f1_scores.append(f1)

        token_f1_scores.append(token_f1)
        _, _, f1_dedup = EvaluationEngine.evaluate_dedup_bertscore(g_texts, p_texts)
        f1_dedup_scores.append(f1_dedup)

    print("Model processing complete!        ", file=sys.stderr)

    df_master["Set_Text_F1"] = set_text_f1_scores
    df_master["BS_F1"] = f1_scores
    df_master["Token_F1"] = token_f1_scores
    df_master["BS_F1_Dedup"] = f1_dedup_scores

    return df_master 

def run_correlation_pipeline():
    df_master = get_evaluation_data()
    if df_master is None: return

    rho_set_text_f1, _ = spearmanr(df_master["Normalized_Human_Score"], df_master["Set_Text_F1"])
    rho_f, _ = spearmanr(df_master["Normalized_Human_Score"], df_master["BS_F1"])
    rho_token_f1, _ = spearmanr(df_master["Normalized_Human_Score"], df_master["Token_F1"])
    rho_f_dedup, _ = spearmanr(df_master["Normalized_Human_Score"], df_master["BS_F1_Dedup"])

    model_summary = df_master.groupby("True_Model_Identity").agg(
        Samples_Evaluated=("Normalized_Human_Score", "count"),
        Avg_Norm_Human_Score=("Normalized_Human_Score", "mean"),
        Avg_Set_Text_F1=("Set_Text_F1", "mean"),
        Avg_BS_F1=("BS_F1", "mean"),
        Avg_Token_F1=("Token_F1", "mean"),
        Avg_BS_F1_Dedup=("BS_F1_Dedup", "mean")
    ).round(3)

    # sorting variables needed for the side-by-side Matrix
    human_sorted = model_summary.sort_values(by="Avg_Norm_Human_Score", ascending=False).reset_index()
    strict_machine_sorted = model_summary.sort_values(by="Avg_BS_F1", ascending=False).reset_index()
    relaxed_machine_sorted = model_summary.sort_values(by="Avg_BS_F1_Dedup", ascending=False).reset_index()

    side_by_side_ranking = []
    for i in range(len(model_summary)):
        side_by_side_ranking.append({
            "Rank": f"#{i+1}",
            "Human Preference": f"{human_sorted.loc[i, 'True_Model_Identity']} ({human_sorted.loc[i, 'Avg_Norm_Human_Score']:.3f})",
            "Strict Machine Preference (F1)": f"{strict_machine_sorted.loc[i, 'True_Model_Identity']} ({strict_machine_sorted.loc[i, 'Avg_BS_F1']:.3f})",
            "Relaxed Machine Preference (Dedup F1)": f"{relaxed_machine_sorted.loc[i, 'True_Model_Identity']} ({relaxed_machine_sorted.loc[i, 'Avg_BS_F1_Dedup']:.3f})"
        })
    df_ranking_matrix = pd.DataFrame(side_by_side_ranking)

    clear_assets_dir()

    # generate charts
    generate_performance_quadrant(
        df=model_summary.reset_index(),
        model_col="True_Model_Identity",
        filename="pilot_quadrant.png"
    )
    
    # print markdown report
    print(f"\n*Total Framework Samples: {len(df_master)}*\n")
    
    print("### Correlation Comparison: Strict vs. Relaxed Metrics\n")
    print("| Evaluation Dimension | Strict Metric | Relaxed Metric | Strict Spearman (ρ) | Relaxed Spearman (ρ) |")
    print("|---|---|---|---|---|")
    print(f"| **Lexical Match** | Set Text F1 | SQuAD Token F1 | {rho_set_text_f1:.4f} | **{rho_token_f1:.4f}** |")
    print(f"| **Semantic Match** | DeBERTa F1 | Deduped DeBERTa F1 | {rho_f:.4f} | **{rho_f_dedup:.4f}** |\n")
    print("\n### Side-by-Side Ranking Comparison\n")
    print(df_ranking_matrix.to_markdown(index=False))
    
    print("\n![Pilot Performance Quadrant](assets/pilot_quadrant.png)\n")

if __name__ == "__main__":
    run_correlation_pipeline()
