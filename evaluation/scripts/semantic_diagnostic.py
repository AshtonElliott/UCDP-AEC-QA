import os
import sys
import glob
import pandas as pd
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
    print("loading human sheets...", file=sys.stderr)
    
    csv_files = glob.glob(os.path.join(COMPLETED_DIR, "*.csv"))
    if not csv_files: return

    all_scores = []
    wide_df = None
    rater_names = []
    consensus_text_df = None

    for file in csv_files:
        df_temp = pd.read_csv(file)
        if 'Human_Score_1_to_5' in df_temp.columns:
            rater_name = os.path.basename(file).split('_')[0]
            rater_names.append(rater_name)
            
            if consensus_text_df is None:
                cols_to_keep = ['Evaluation_ID', 'Source_Article', 'Human_Ground_Truth', 'LLM_Prediction']
                cols_to_keep = [c for c in cols_to_keep if c in df_temp.columns]
                consensus_text_df = df_temp[cols_to_keep].copy()
            
            clean = df_temp[['Evaluation_ID', 'Human_Score_1_to_5']].dropna()
            all_scores.append(clean)
            
            df_subset = clean.rename(columns={'Human_Score_1_to_5': rater_name})
            if wide_df is None:
                wide_df = df_subset
            else:
                wide_df = pd.merge(wide_df, df_subset, on='Evaluation_ID', how='inner')

    df_template = pd.read_csv(MASTER_TEMPLATE)
    df_key = pd.read_csv(KEY_FILE)
    df_merged = pd.merge(df_template, df_key, on="Evaluation_ID", how="left")
    df_merged = pd.merge(df_merged, wide_df, on="Evaluation_ID", how="inner")
    
    if consensus_text_df is not None:
        cols_to_drop = [c for c in consensus_text_df.columns if c != 'Evaluation_ID' and c in df_merged.columns]
        df_merged = df_merged.drop(columns=cols_to_drop)
        df_merged = pd.merge(df_merged, consensus_text_df, on="Evaluation_ID", how="inner")

    null_identifiers = ["", "nan", "NO ANSWER"]
    complex_mask = ~df_merged["Human_Ground_Truth"].astype(str).str.strip().isin(null_identifiers)
    df_complex = df_merged[complex_mask].copy().reset_index(drop=True)

    if len(df_complex) == 0: return

    df_all_raters = pd.concat(all_scores)
    df_all_raters = df_all_raters[df_all_raters['Evaluation_ID'].isin(df_complex['Evaluation_ID'])]
    df_consolidated = df_all_raters.groupby('Evaluation_ID', as_index=False).agg(
        Consolidated_Human_Score=('Human_Score_1_to_5', 'mean')
    )
    df_complex = pd.merge(df_complex, df_consolidated, on='Evaluation_ID', how='inner')
    df_complex["Normalized_Human_Score"] = (df_complex["Consolidated_Human_Score"] - 1) / 4

    print("\ncalculating IAA...", file=sys.stderr)
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
    except Exception:
        complex_alpha = float('nan')
    
    icc_results = pg.intraclass_corr(data=long_df, targets='Evaluation_ID', raters='Rater', ratings='Score')
    match = icc_results[icc_results['Type'] == 'ICC(A,k)']
    complex_icc = match['ICC'].values[0] if not match.empty else float('nan')

    print("running evaluation metrics...", file=sys.stderr)
    
    set_text_f1_scores = []
    f1_scores = []
    token_f1_scores = []
    
    for idx, row in df_complex.iterrows():
        print(f"processing article {idx + 1}/{len(df_complex)}...", end="\r", file=sys.stderr)
        
        human_str = str(row.get("Human_Ground_Truth", "")).strip()
        llm_str = str(row.get("LLM_Prediction", "")).strip()

        g_texts = [] if human_str in null_identifiers else [s.strip() for s in human_str.split(" | ") if s.strip()]
        p_texts = [] if llm_str in null_identifiers else [s.strip() for s in llm_str.split(" | ") if s.strip()]

        set_text_f1, token_f1 = EvaluationEngine.evaluate_ie_squad_metrics(g_texts, p_texts)
        set_text_f1_scores.append(set_text_f1)
        _, _, bertscore_f1 = EvaluationEngine.evaluate_bipartite_bertscore(g_texts, p_texts)
        f1_scores.append(bertscore_f1)
        
        token_f1_scores.append(token_f1)
        
    print("Model processing complete!        ", file=sys.stderr)

    df_complex["Set_Text_F1"] = set_text_f1_scores
    df_complex["BS_F1"] = f1_scores
    df_complex["Token_F1"] = token_f1_scores
    
    df_complex["Score_Delta"] = abs(df_complex["Normalized_Human_Score"] - df_complex["BS_F1"])
    df_disagreements = df_complex.sort_values(by="Score_Delta", ascending=False).head(analysis_num)

    rho_set_text_f1, _ = spearmanr(df_complex["Normalized_Human_Score"], df_complex["Set_Text_F1"])
    rho_f, _ = spearmanr(df_complex["Normalized_Human_Score"], df_complex["BS_F1"])
    rho_token_f1, _ = spearmanr(df_complex["Normalized_Human_Score"], df_complex["Token_F1"])

    print(f"\n*Evaluated on {len(df_complex)} complex edge-case rows.*\n")
    
    print("### Human Baseline Agreement (Complex Subset)\n")
    print("| Metric | Score |")
    print("|---|---|")
    print(f"| **Krippendorff's Alpha** | {complex_alpha:.4f} |")
    print(f"| **Intraclass Correlation ICC(2,k)** | {complex_icc:.4f} |\n")
    
    print("### Strict vs. Relaxed Metric Correlation\n")
    print("| Evaluation Dimension | Strict Metric | Relaxed Metric | Strict Spearman (ρ) | Relaxed Spearman (ρ) |")
    print("|---|---|---|---|---|")
    print(f"| **Lexical Match** | Span F1 | SQuAD Token F1 | {rho_set_text_f1:.4f} | {rho_token_f1:.4f} |")
    print(f"| **Semantic Match** | BERTScore | BERTScore | {rho_f:.4f} | {rho_f:.4f} |\n")
    
    print("### Delta Analysis (Top 5 Disagreements after Relaxation)\n")
    for i, (_, row) in enumerate(df_disagreements.iterrows(), 1):
        print(f"**Disagreement #{i} (Score Delta: {row['Score_Delta']:.4f})**")
        print(f"* **ID:** {row['Evaluation_ID']}")
        print(f"* **Ground Truth:** `{row['Human_Ground_Truth']}`")
        print(f"* **AI Prediction:** `{row['LLM_Prediction']}`")
        print(f"* **Scores:** Human = {row['Normalized_Human_Score']:.4f} | Relaxed BERTScore = {row['BS_F1_Dedup']:.4f}\n")
        print("---\n")

if __name__ == "__main__":
    run_comprehensive_evaluation()