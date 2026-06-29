import os
import json
import random
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) if '__file__' in locals() else os.getcwd()
DATA_DIR = os.path.join(BASE_DIR, 'data')
RAW_DIR = os.path.join(DATA_DIR, 'raw_inputs')
TEMPLATES_DIR = os.path.join(DATA_DIR, 'grading_templates')
RESULTS_DIR = os.path.join(DATA_DIR, 'evaluation_results')

for directory in [TEMPLATES_DIR, RESULTS_DIR]:
    os.makedirs(directory, exist_ok=True)

GT_FILE = os.path.join(RAW_DIR, 'train.json') 

LLM_FILES = {
    "Gemma 3.4B": os.path.join(RAW_DIR, "gemma3.4b_results.json"),
    "Gemma 4.e4B": os.path.join(RAW_DIR, "gemma4.e4b_results.json"),
    "Llama 3.1": os.path.join(RAW_DIR, "llama3.1_Results.json"),
    "Mistral": os.path.join(RAW_DIR, "mistral_Results.json"),
    "Qwen 3.8B": os.path.join(RAW_DIR, "qwen3.8b_Results.json")
}

SAMPLE_SIZE = 100

def load_json_file(filepath):
    if not os.path.exists(filepath):
        print(f"Warning: File not found at {filepath}")
        return []
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)

def build_article_map(json_data):
    mapping = {}
    for entry in json_data:
        article_text = entry.get('source_article', '').strip()
        if article_text:
            mapping[article_text] = entry
    return mapping

def generate_blind_pilot():
    print("loading GT data...")
    gt_data = load_json_file(GT_FILE)
    if not gt_data: return
        
    gt_map = build_article_map(gt_data)
    
    # stratified sampling: 50/50 split between articles with answers and those without
    print("stratifying data...")
    articles_with_answers = []
    articles_without_answers = []

    for article, entry in gt_map.items():
        labels = entry.get('answer_labels', [])
        is_no_answer = entry.get('no_answer', False)

        # if there are no labels or it's explicitly flagged as no_answer
        if not labels or is_no_answer:
            articles_without_answers.append(article)
        else:
            articles_with_answers.append(article)

    if len(articles_with_answers) + len(articles_without_answers) < SAMPLE_SIZE:
        print(f"error: not enough total articles for {SAMPLE_SIZE} articles, found {len(articles_with_answers) + len(articles_without_answers)}")
        return

    print("loading LLM inference files...")
    llm_maps = {}
    for model_name, path in LLM_FILES.items():
        llm_data = load_json_file(path)
        if llm_data:
            llm_maps[model_name] = build_article_map(llm_data)

    active_models = list(llm_maps.keys())
    random.shuffle(active_models)
    
    # calculate exact split 
    target_complex = SAMPLE_SIZE // 2
    
    # edge case when one bucket has fewer articles than the target
    num_complex = min(target_complex, len(articles_with_answers))
    num_null = SAMPLE_SIZE - num_complex

    sampled_complex = random.sample(articles_with_answers, num_complex)
    sampled_null = random.sample(articles_without_answers, num_null)

    # combine and shuffle
    sampled_articles = sampled_complex + sampled_null
    random.shuffle(sampled_articles)
    
    allocations = {}
    for i, article in enumerate(sampled_articles):
        assigned_model = active_models[i % len(active_models)]
        allocations[article] = assigned_model

    blind_grading_rows = []
    decryption_key_rows = []

    print("generating sheets...")
    for idx, article in enumerate(sampled_articles, 1):
        assigned_model = allocations[article]
        
        gt_entry = gt_map[article]
        gt_spans = gt_entry.get('answer_labels', [])
        human_text = " | ".join([g.get('text', '') for g in gt_spans]).strip() if gt_spans else "NO ANSWER"

        model_entry = llm_maps[assigned_model].get(article, {})
        pred_spans = model_entry.get('answer_labels', model_entry.get('model_spans', []))
        
        if model_entry.get('no_answer') is True or not pred_spans:
            llm_text = "NO ANSWER"
        else:
            llm_text = " | ".join([p.get('text', '') for p in pred_spans]).strip()

        blind_grading_rows.append({
            "Evaluation_ID": f"EVAL_{idx:03d}",
            "Source_Article": article,
            "Human_Ground_Truth": human_text,
            "LLM_Prediction": llm_text,
            "Human_Score_1_to_5": "" 
        })

        decryption_key_rows.append({
            "Evaluation_ID": f"EVAL_{idx:03d}",
            "True_Model_Identity": assigned_model
        })

    df_blind = pd.DataFrame(blind_grading_rows)
    df_key = pd.DataFrame(decryption_key_rows)

    # export the template
    out_path = os.path.join(TEMPLATES_DIR, "master_grading_template.csv")
    df_blind.to_csv(out_path, index=False, encoding='utf-8')

    # save the decryption key
    key_out = os.path.join(RESULTS_DIR, "secret_decryption_key.csv")
    df_key.to_csv(key_out, index=False, encoding='utf-8')


    print("="*50)
    print(f"total Samples: {SAMPLE_SIZE} ({num_complex} Complex | {num_null} Null)")
    print(f"grading template saved to: {out_path}")
    print(f"decryption key saved to: {key_out}")
    print("="*50)

if __name__ == "__main__":
    generate_blind_pilot()