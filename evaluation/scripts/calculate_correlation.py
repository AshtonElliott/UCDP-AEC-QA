import os
import glob
import pandas as pd
import warnings
from scipy.stats import spearmanr

from scripts.core import EvaluationEngine

os.environ["TOKENIZERS_PARALLELISM"] = "false" 
warnings.filterwarnings("ignore")

SCRIPT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) if '__file__' in locals() else os.getcwd()
DATA_DIR = os.path.join(SCRIPT_DIR, 'data')
COMPLETED_DIR = os.path.join(DATA_DIR, 'completed_grading')
MASTER_TEMPLATE = os.path.join(DATA_DIR, "grading_templates", "master_grading_template.csv")
KEY_FILE = os.path.join(DATA_DIR, "evaluation_results", "secret_decryption_key.csv")

def get_evaluation_data():
    print("loading human rater sheets...")
    csv_files = glob.glob(os.path.join(COMPLETED_DIR, "*.csv"))
    if not csv_files:
        print(f"Error: No completed grading sheets found in {COMPLETED_DIR}")
        return None

    # average all human scores
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
        print("error: could not find valid 'Human_Score_1_to_5' columns in the rater sheets")
        return None

    df_all_raters = pd.concat(all_scores)
    df_consolidated = df_all_raters.groupby('Evaluation_ID', as_index=False).agg(
        Consolidated_Human_Score=('Human_Score_1_to_5', 'mean')
    )
    
    # merge with the original template to get the model identity and predictions
    if not os.path.exists(MASTER_TEMPLATE) or not os.path.exists(KEY_FILE):
        print("error: missing master_grading_template.csv or secret_decryption_key.csv")
        return None
        
    df_template = pd.read_csv(MASTER_TEMPLATE)
    df_key = pd.read_csv(KEY_FILE)

    df_master = pd.merge(df_template, df_consolidated, on="Evaluation_ID", how="inner")
    df_master = pd.merge(df_master, df_key, on="Evaluation_ID", how="inner")
    
    # normalization: convert 1-5 human scale to 0-1 scale
    df_master["Normalized_Human_Score"] = (df_master["Consolidated_Human_Score"] - 1) / 4

    precisions = []
    recalls = []
    f1_scores = []
    squad_em_scores = []
    
    for idx, row in df_master.iterrows():
        print(f"Processing article {idx + 1}/{len(df_master)}...", end="\r")
        
        human_str = str(row.get("Human_Ground_Truth", "")).strip()
        llm_str = str(row.get("LLM_Prediction", "")).strip()

        g_texts = [] if human_str in ["", "nan", "NO ANSWER"] else [s.strip() for s in human_str.split(" | ") if s.strip()]
        p_texts = [] if llm_str in ["", "nan", "NO ANSWER"] else [s.strip() for s in llm_str.split(" | ") if s.strip()]

        # 1. SQuAD EM calculation via Core Engine
        squad_em = EvaluationEngine.evaluate_exact_match(g_texts, p_texts)
        squad_em_scores.append(squad_em)

        # 2. Bipartite BERTScore calculation via Core Engine
        p, r, f1 = EvaluationEngine.evaluate_bipartite_bertscore(g_texts, p_texts)
        precisions.append(p)
        recalls.append(r)
        f1_scores.append(f1)

    # Clear the processing line
    print("✅ Model processing complete!                             ")

    df_master["BS_Precision"] = precisions
    df_master["BS_Recall"] = recalls
    df_master["BS_F1"] = f1_scores
    df_master["SQuAD_EM"] = squad_em_scores

    return df_master 

def run_correlation_pipeline():
    df_master = get_evaluation_data()
    if df_master is None:
        return

    # spearman rank matrix
    rho_squad, p_val_squad = spearmanr(df_master["Normalized_Human_Score"], df_master["SQuAD_EM"])
    rho_f, p_val_f = spearmanr(df_master["Normalized_Human_Score"], df_master["BS_F1"])
    rho_p, p_val_p = spearmanr(df_master["Normalized_Human_Score"], df_master["BS_Precision"])
    rho_r, p_val_r = spearmanr(df_master["Normalized_Human_Score"], df_master["BS_Recall"])

    # summary metrics by model
    model_summary = df_master.groupby("True_Model_Identity").agg(
        Samples_Evaluated=("Normalized_Human_Score", "count"),
        Avg_Norm_Human_Score=("Normalized_Human_Score", "mean"),
        Avg_SQuAD_EM=("SQuAD_EM", "mean"),
        Avg_BS_F1=("BS_F1", "mean"),
        Avg_BS_Precision=("BS_Precision", "mean"),
        Avg_BS_Recall=("BS_Recall", "mean")
    ).round(3)
    
    # table 1: sorted model performance summary
    model_summary_sorted = model_summary.sort_values(by="Avg_BS_F1", ascending=False).copy()
    model_summary_sorted.insert(0, "Rank", range(1, len(model_summary_sorted) + 1))

    # table 2: side-by-side ranking comparison
    human_sorted = model_summary.sort_values(by="Avg_Norm_Human_Score", ascending=False).reset_index()
    machine_sorted = model_summary.sort_values(by="Avg_BS_F1", ascending=False).reset_index()

    side_by_side_ranking = []
    for i in range(len(model_summary)):
        side_by_side_ranking.append({
            "Rank": f"#{i+1}",
            "Human Expert Preference": f"{human_sorted.loc[i, 'True_Model_Identity']} ({human_sorted.loc[i, 'Avg_Norm_Human_Score']:.3f})",
            "Machine Metric Preference (Avg_BS_F1)": f"{machine_sorted.loc[i, 'True_Model_Identity']} ({machine_sorted.loc[i, 'Avg_BS_F1']:.3f})"
        })
    df_ranking_matrix = pd.DataFrame(side_by_side_ranking)

    # display the results
    print("="*95)
    print(f"Total Framework Samples: {len(df_master)}")        
    print("Comparison Table (Spearman Rank Matrix):")
    print("-" * 95)
    print(f"  Evaluation Dimension | Spearman rho (ρ) | p-value")
    print("-" * 95)
    print(f"  SQuAD Exact Match    | {rho_squad:16.4f} | {p_val_squad:.5e}")
    print(f"  DeBERTa Precision    | {rho_p:16.4f} | {p_val_p:.5e}")
    print(f"  DeBERTa Recall       | {rho_r:16.4f} | {p_val_r:.5e}")
    print(f"  DeBERTa F1-Score     | {rho_f:16.4f} | {p_val_f:.5e}")
    print("-" * 95)
    
    print("\nLLM Comparison Summary (Sorted by Avg_BS_F1):")
    print("-" * 95)
    print(model_summary_sorted.to_string(index=True))
    print("-" * 95)
    
    print("\nSide-by-Side Ranking Comparison:")
    print("-" * 95)
    print(df_ranking_matrix.to_string(index=False))
    print("="*95)

if __name__ == "__main__":
    run_correlation_pipeline()