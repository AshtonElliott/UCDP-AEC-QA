import os
import json
import re
import glob
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONSENSUS_FILE = os.path.join(BASE_DIR, "data", "raw_inputs", "ground_truth.json")
MASTER_TEMPLATE = os.path.join(BASE_DIR, "data", "grading_templates", "master_grading_template.csv")
MISMATCH_REPORT = os.path.join(BASE_DIR, "data", "grading_templates", "mismatched_articles.csv")
COMPLETED_DIR = os.path.join(BASE_DIR, "data", "completed_grading")

def sanitize_for_key(text):
    # strip everything except letters/numbers, lower it. 
    return re.sub(r'\W+', '', str(text)).lower()

def normalize_and_sort_tokens(gt_str):
    clean_str = str(gt_str).strip()
    if clean_str.upper() in ["NO ANSWER", "ARTICLE_NOT_FOUND_IN_CONSENSUS", "NAN", ""]:
        return [clean_str.upper()]
    
    tokens = clean_str.split("|")
    cleaned_tokens = []
    for t in tokens:
        # strip all punctuation out of the token itself, lower it, and fix spacing
        t_clean = re.sub(r'[^\w\s]', '', t).strip().lower()
        t_clean = re.sub(r'\s+', ' ', t_clean) # collapse double spaces
        if t_clean:
            cleaned_tokens.append(t_clean)
            
    return sorted(cleaned_tokens)

def check_ground_truth_deltas():
    with open(CONSENSUS_FILE, 'r', encoding='utf-8') as f:
        consensus_data = json.load(f)
        
    consensus_map = {}
    for entry in consensus_data:
        article_text = entry.get("source_article", entry.get("text", ""))
        
        if entry.get("no_answer"):
            consensus_str = "NO ANSWER"
        else:
            labels = entry.get("answer_labels", [])
            consensus_str = " | ".join([lbl.get("text", "") for lbl in labels]).strip()
            if not consensus_str:
                consensus_str = "NO ANSWER"
                
        lookup_key = sanitize_for_key(article_text)
        consensus_map[lookup_key] = consensus_str

    df = pd.read_csv(MASTER_TEMPLATE)
    changed_rows = []
    
    to_drop_ids = []
    to_regrade_dict = {}
    
    missing_articles = 0

    for index, row in df.iterrows():
        raw_article = row["Source_Article"]
        old_gt = str(row["Human_Ground_Truth"]).strip()
        eval_id = row["Evaluation_ID"]
        
        lookup_key = sanitize_for_key(raw_article)
        new_gt = consensus_map.get(lookup_key, "ARTICLE_NOT_FOUND_IN_CONSENSUS")
        
        if new_gt == "ARTICLE_NOT_FOUND_IN_CONSENSUS":
            missing_articles += 1
            # track the ID to drop it later, then skip the array printout
            to_drop_ids.append(eval_id)
            continue 
        
        normalized_old = normalize_and_sort_tokens(old_gt)
        normalized_new = normalize_and_sort_tokens(new_gt)
        
        if normalized_old != normalized_new:
            # ADJUSTED: Keep the master template layout format for peer review sheets
            changed_rows.append({
                "Evaluation_ID": eval_id,
                "Source_Article": raw_article,
                "Human_Ground_Truth": new_gt, # Swapped with the correct consensus value
                "LLM_Prediction": row.get("LLM_Prediction", ""),
                "Human_Score_1_to_5": "" # Cleared slate for your peers to grade
            })
            # track the ID and the new GT to update it later
            to_regrade_dict[eval_id] = new_gt
            
            print(f"\nID: {eval_id}")
            print(f"Old Array : {normalized_old}")
            print(f"New Array : {normalized_new}")

    if changed_rows:
        df_mismatches = pd.DataFrame(changed_rows)
        # Ensure the output explicitly mirrors the requested columns order
        column_order = ["Evaluation_ID", "Source_Article", "Human_Ground_Truth", "LLM_Prediction", "Human_Score_1_to_5"]
        df_mismatches = df_mismatches[column_order]
        df_mismatches.to_csv(MISMATCH_REPORT, index=False, encoding='utf-8')

    print("\n" + "=" * 70)
    print(f"Total checked: {len(df)}")
    print(f"Articles completely failing to link: {missing_articles}")
    print(f"Articles with mismatched tokens: {len(changed_rows)}")
    print("=" * 70)

    # clean up the grading sheets in the completed directory
    grader_files = glob.glob(os.path.join(COMPLETED_DIR, "*.csv"))
    
    if not grader_files:
        print(f"\nWarning: No grading sheets found in {COMPLETED_DIR} to update.")
    else:
        print("\nUpdating Peer Grading Sheets...")
        for file in grader_files:
            file_name = os.path.basename(file)
            df_grader = pd.read_csv(file)
            original_len = len(df_grader)
            
            # delete the conflicted/missing rows completely
            if to_drop_ids:
                df_grader = df_grader[~df_grader["Evaluation_ID"].isin(to_drop_ids)]
            
            # update GT and clear old scores for regrading
            if "Human_Ground_Truth" in df_grader.columns and "Human_Score_1_to_5" in df_grader.columns:
                for eid, updated_gt in to_regrade_dict.items():
                    df_grader.loc[df_grader["Evaluation_ID"] == eid, "Human_Ground_Truth"] = updated_gt
                    df_grader.loc[df_grader["Evaluation_ID"] == eid, "Human_Score_1_to_5"] = pd.NA
            
            # save the file back over itself
            df_grader.to_csv(file, index=False, encoding='utf-8')
            
            dropped_count = original_len - len(df_grader)
            print(f"  -> Processed {file_name}: Dropped {dropped_count} rows, Erased scores for {len(to_regrade_dict)} mismatched rows.")
        print("=" * 70)

if __name__ == "__main__":
    check_ground_truth_deltas()