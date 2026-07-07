import os
import glob
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
COMPLETED_DIR = os.path.join(BASE_DIR, "data", "completed_grading")

def merge_peer_specific_scores():
    # find all re-graded sheets ending with '_mismatched_graded.csv'
    regraded_pattern = os.path.join(COMPLETED_DIR, "*_mismatched_graded.csv")
    regraded_files = glob.glob(regraded_pattern)
    
    if not regraded_files:
        print(f"Error: No re-graded files found matching '*_mismatched_graded.csv' in {COMPLETED_DIR}")
        return

    # process each re-graded file
    for regraded_file in regraded_files:
        regraded_name = os.path.basename(regraded_file)
        
        rater_prefix = regraded_name.split('_mismatched_graded')[0]
        
        # find the original matching sheet in the same directory
        original_pattern = os.path.join(COMPLETED_DIR, f"{rater_prefix}_*.csv")
        all_rater_files = glob.glob(original_pattern)
        original_files = [f for f in all_rater_files if "_mismatched_graded.csv" not in f]
        
        if not original_files:
            print(f"Could not find original sheet for rater '{rater_prefix}' (Expected '{rater_prefix}_*.csv')")
            continue
            
        original_file = original_files[0] 
        original_name = os.path.basename(original_file)
        
        # load both datasets
        df_regraded = pd.read_csv(regraded_file)
        df_original = pd.read_csv(original_file)
        
        # clean out any empty rows in the newly graded sheets
        df_regraded_clean = df_regraded.dropna(subset=["Human_Score_1_to_5"])
        score_map = dict(zip(df_regraded_clean["Evaluation_ID"], df_regraded_clean["Human_Score_1_to_5"]))
        
        if not score_map:
            print(f"No valid scores found inside {regraded_name}. Skipping.")
            continue
            
        # merge the scores row-by-row into the original template
        updated_count = 0
        if "Evaluation_ID" in df_original.columns and "Human_Score_1_to_5" in df_original.columns:
            for index, row in df_original.iterrows():
                eid = row["Evaluation_ID"]
                if eid in score_map:
                    df_original.at[index, "Human_Score_1_to_5"] = score_map[eid]
                    updated_count += 1
            
            # overwrite the original file with the newly restored rows
            df_original.to_csv(original_file, index=False, encoding='utf-8')
            print(f"Merged {updated_count} scores from {regraded_name} -> {original_name}")
        else:
            print(f"Columns missing in original file {original_name}. Skipping.")

    print("-" * 70)
    print("All individual sheets successfully merged!")
    print("=" * 70)

if __name__ == "__main__":
    merge_peer_specific_scores()