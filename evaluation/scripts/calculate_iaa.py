import os
import glob
import pandas as pd
import pingouin as pg
import nltk
from nltk.metrics.agreement import AnnotationTask

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) if '__file__' in locals() else os.getcwd()
COMPLETED_DIR = os.path.join(BASE_DIR, 'data', 'completed_grading')

def calculate_advanced_iaa():
    csv_files = glob.glob(os.path.join(COMPLETED_DIR, "*.csv"))
    
    if not csv_files:
        print(f"error: no completed grading sheets found in {COMPLETED_DIR}")
        return

    print(f"found {len(csv_files)} grading sheets. Merging...")
    
    wide_df = None
    rater_names = []
    long_data = []
    nltk_data = []

    for file in csv_files:
        df = pd.read_csv(file)
        
        if 'Human_Score_1_to_5' not in df.columns:
            continue
            
        rater_name = os.path.basename(file).split('_')[0]
        rater_names.append(rater_name)
            
        df_subset = df[['Evaluation_ID', 'Human_Score_1_to_5']].rename(
            columns={'Human_Score_1_to_5': rater_name}
        )
        
        if wide_df is None:
            wide_df = df_subset
        else:
            wide_df = pd.merge(wide_df, df_subset, on='Evaluation_ID', how='outer')
            
        for _, row in df.iterrows():
            if pd.notna(row['Human_Score_1_to_5']):
                score = float(row['Human_Score_1_to_5'])
                item_id = str(row['Evaluation_ID']).strip() 
                
                long_data.append({
                    'Evaluation_ID': item_id,
                    'Rater': rater_name,
                    'Score': score
                })
                nltk_data.append((rater_name, item_id, score))

    # clean the wide dataframe
    wide_df = wide_df.dropna(subset=rater_names)
    long_df = pd.DataFrame(long_data)
    
    # ensure IDs match 
    valid_ids = set(wide_df['Evaluation_ID'].astype(str))
    long_df = long_df[long_df['Evaluation_ID'].isin(valid_ids)]
    nltk_data = [item for item in nltk_data if item[1] in valid_ids]

    if len(wide_df) == 0:
        print("error: no overlapping fully graded articles found.")
        return

    print(f"calculating metrics across {len(wide_df)} graded articles...\n")
    
    # calculate krippendorff's alpha
    task = AnnotationTask(data=nltk_data, distance=lambda x, y: (x - y)**2)
    try:
        k_alpha = task.alpha()
    except Exception as e:
        k_alpha = float('nan')
        print(f"Alpha calculation error: {e}")
    
    # calculate ICC(2,k) absolute agreement 
    icc_results = pg.intraclass_corr(
        data=long_df, 
        targets='Evaluation_ID', 
        raters='Rater', 
        ratings='Score'
    )
    
    icc2k_score = "ERROR (See raw output below)"
    
    # diagnostic check: try to extract the score 
    if isinstance(icc_results, pd.DataFrame) and not icc_results.empty:
        if 'Type' in icc_results.columns:
            match = icc_results[icc_results['Type'] == 'ICC(A,k)']
            if not match.empty:
                icc2k_score = f"{match['ICC'].values[0]:.4f}"
            else:
                fallback = icc_results[icc_results['Type'] == 'ICC2']
                if not fallback.empty:
                    icc2k_score = f"{fallback['ICC'].values[0]:.4f} (Using ICC2 fallback)"
        
    # output report
    print("="*65)
    print(f"Total Overlapping Articles: {len(wide_df)}")    
    print(f"1. Krippendorff's Alpha (Ordinal):   {k_alpha:.4f}")
    print(f"2. ICC(2,k) (Absolute Agreement):    {icc2k_score}")
    print("-" * 65)
    
    # if ICC extraction failed, print the raw table to debug
    if "ERROR" in icc2k_score:
        print("raw output:")
        print(icc_results)
        
    print("="*65)

if __name__ == "__main__":
    calculate_advanced_iaa()

