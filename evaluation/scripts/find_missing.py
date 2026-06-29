import pandas as pd
import glob
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) if '__file__' in locals() else os.getcwd()
COMPLETED_DIR = os.path.join(BASE_DIR, 'data', 'completed_grading')

def find_missing_rows():
    files = glob.glob(os.path.join(COMPLETED_DIR, "*.csv"))
    
    if not files:
        print("No CSV files found.")
        return

    for file in files:
        df = pd.read_csv(file)
        
        # look for any rows where the score is blank (NaN)
        missing = df[df['Human_Score_1_to_5'].isna()]
        
        if len(missing) > 0:
            rater_name = os.path.basename(file)
            print("-" * 50)
            print(f"Rater {rater_name} missed {len(missing)} articles:")
            # print the exact IDs they skipped
            print(missing['Evaluation_ID'].tolist())

if __name__ == "__main__":
    find_missing_rows()