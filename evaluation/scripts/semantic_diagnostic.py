import os
import glob
import pandas as pd
import numpy as np
import pingouin as pg
from scipy.stats import spearmanr
from nltk.metrics.agreement import AnnotationTask
import warnings

from scripts.core import EvaluationEngine

os.environ["TOKENIZERS_PARALLELISM"] = "false" 
warnings.filterwarnings("ignore")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) if '__file__' in locals() else os.getcwd()
COMPLETED_DIR = os.path.join(BASE_DIR, 'data', 'completed_grading')
MASTER_TEMPLATE = os.path.join(BASE_DIR, 'data', "grading_templates", "master_grading_template.csv")
KEY_FILE = os.path.join(BASE_DIR, 'data', "evaluation_results", "secret_decryption_key.csv")

analysis_num = 5

def run_comprehensive_evaluation():
    print("loading human sheets...")
    
    csv_files = glob.glob(os.path.join(COMPLETED_DIR, "*.csv"))
    if not csv_files:
        print(f"error: no completed grading sheets found in {COMPLETED_DIR}")
        return

    all_scores = []
    wide_df = None
    rater_names = []

    for file in csv_files:
        df_temp = pd.read_csv(file)
        if 'Human_Score_1_to_5' in df_temp.columns:
            rater_name = os.path.basename(file).split('_')[0]
            rater_names.append(rater_name)
            
            clean = df_temp[['Evaluation_ID', 'Human_Score_1_to_5']].dropna()
            all_scores.append(clean)
            
            df_subset = clean.rename(columns={'Human_Score_1_to_5': rater_name})
            if wide_df is None:
                wide_df = df_subset
            else:
                wide_df = pd.merge(wide_df, df_subset, on='Evaluation_ID', how='inner')

    # merge with templates
    df_template = pd.read_csv(MASTER_TEMPLATE)
    df_key = pd.read_csv(KEY_FILE)
    
    df_merged = pd.merge(df_template, wide_df, on="Evaluation_ID", how="inner")
    df_merged = pd.merge(df_merged, df_key, on="Evaluation_ID", how="inner")

    # filter by GT
    null_identifiers = ["", "nan", "NO ANSWER"]
    complex_mask = ~df_merged["Human_Ground_Truth"].astype(str).str.strip().isin(null_identifiers)
    df_complex = df_merged[complex_mask].copy()
    df_complex = df_complex.reset_index(drop=True)

    if len(df_complex) == 0:
        print("Error: No complex-only rows found after filtering.")
        return

    print(f"isolated {len(df_complex)} rows.")

    df_all_raters = pd.concat(all_scores)
    df_all_raters = df_all_raters[df_all_raters['Evaluation_ID'].isin(df_complex['Evaluation_ID'])]
    df_consolidated = df_all_raters.groupby('Evaluation_ID', as_index=False).agg(
        Consolidated_Human_Score=('Human_Score_1_to_5', 'mean')
    )
    df_complex = pd.merge(df_complex, df_consolidated, on='Evaluation_ID', how='inner')
    df_complex["Normalized_Human_Score"] = (df_complex["Consolidated_Human_Score"] - 1) / 4

    # recalculate human IAA
    print("\ncalculating IAA...")
    long_data = []
    nltk_data = []
    for _, row in df_complex.iterrows():
        item_id = str(row['Evaluation_ID']).strip()
        for rater in rater_names:
            score_val = float(row[rater])
            long_data.append({'Evaluation_ID': item_id, 'Rater': rater, 'Score': score_val})
            nltk_data.append((rater, item_id, score_val))
            
    long_df = pd.DataFrame(long_data)
    task = AnnotationTask(data=nltk_data, distance=lambda x, y: (x - y)**2)
    
    try:
        complex_alpha = task.alpha()
    except Exception as e:
        complex_alpha = float('nan')
    
    icc_results = pg.intraclass_corr(data=long_df, targets='Evaluation_ID', raters='Rater', ratings='Score')
    match = icc_results[icc_results['Type'] == 'ICC(A,k)']
    complex_icc = match['ICC'].values[0] if not match.empty else float('nan')

    # EM (1-to-1 bipartite + SQuAD)
    print("running evaluation metrics...\n")
    
    precisions = []
    recalls = []
    f1_scores = []
    squad_em_scores = []
    
    for idx, row in df_complex.iterrows():
        print(f"processing article {idx + 1}/{len(df_complex)}...", end="\r")
        
        human_str = str(row.get("Human_Ground_Truth", "")).strip()
        llm_str = str(row.get("LLM_Prediction", "")).strip()

        g_texts = [] if human_str in null_identifiers else [s.strip() for s in human_str.split(" | ") if s.strip()]
        p_texts = [] if llm_str in null_identifiers else [s.strip() for s in llm_str.split(" | ") if s.strip()]

        # calculate SQuAD EM 
        squad_em = EvaluationEngine.evaluate_exact_match(g_texts, p_texts)
        squad_em_scores.append(squad_em)

        # calculate bipartite BERTScore 
        p, r, f1 = EvaluationEngine.evaluate_bipartite_bertscore(g_texts, p_texts)
        precisions.append(p)
        recalls.append(r)
        f1_scores.append(f1)
        
    print("Model processing complete!")

    df_complex["BS_Precision"] = precisions
    df_complex["BS_Recall"] = recalls
    df_complex["BS_F1"] = f1_scores
    df_complex["SQuAD_EM"] = squad_em_scores
    
    # delta analysis
    df_complex["Score_Delta"] = abs(df_complex["Normalized_Human_Score"] - df_complex["BS_F1"])
    df_disagreements = df_complex.sort_values(by="Score_Delta", ascending=False).head(analysis_num)

    # calculate spearman correlations
    rho_p, p_val_p = spearmanr(df_complex["Normalized_Human_Score"], df_complex["BS_Precision"])
    rho_r, p_val_r = spearmanr(df_complex["Normalized_Human_Score"], df_complex["BS_Recall"])
    rho_f, p_val_f = spearmanr(df_complex["Normalized_Human_Score"], df_complex["BS_F1"])
    rho_squad, p_val_squad = spearmanr(df_complex["Normalized_Human_Score"], df_complex["SQuAD_EM"])

    # display outputs
    print("\n" + "="*80)
    print(f"Row Number: {len(df_complex)}\n")
    print(f"Human Baseline Agreement:")
    print(f"  • Krippendorff's Alpha:           {complex_alpha:.4f}")
    print(f"  • Intraclass Correlation ICC(2,k): {complex_icc:.4f}\n")
    
    print(f"Human-Model Comparison:")
    print("-" * 80)
    print(f"  Metric                     | Spearman rho (ρ) | p-value")
    print("-" * 80)
    print(f"  SQuAD Exact Match (Strict) | {rho_squad:16.4f} | {p_val_squad:.5e}")
    print(f"  DeBERTa F1 (Semantic)      | {rho_f:16.4f} | {p_val_f:.5e}")
    print(f"  DeBERTa Precision          | {rho_p:16.4f} | {p_val_p:.5e}")
    print(f"  DeBERTa Recall             | {rho_r:16.4f} | {p_val_r:.5e}")
    print("-" * 80)
    
    print("\n" + "="*80)
    print(" Delta Analysis (Top 5 Disagreements):")
    print("="*80)
    
    for i, (_, row) in enumerate(df_disagreements.iterrows(), 1):
        print(f"\n--- Disagreement #{i} (Score Delta: {row['Score_Delta']:.4f}) ---")
        print(f"ID           : {row['Evaluation_ID']}")
        print(f"Ground Truth : {row['Human_Ground_Truth']}")
        print(f"AI Prediction: {row['LLM_Prediction']}")
        print(f"Scores       : Human = {row['Normalized_Human_Score']:.4f} | BERTScore = {row['BS_F1']:.4f}")
    print("\n" + "="*80 + "\n")

if __name__ == "__main__":
    run_comprehensive_evaluation()