import os
import json
import pandas as pd
import warnings

from scripts.core import EvaluationEngine 

warnings.filterwarnings("ignore")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) if '__file__' in locals() else os.getcwd()
DATA_DIR = os.path.join(BASE_DIR, 'data')
RAW_DIR = os.path.join(DATA_DIR, 'raw_inputs')
GT_FILE = os.path.join(RAW_DIR, 'train.json') 

LLM_FILES = {
    "Gemma 3.4B": os.path.join(RAW_DIR, "gemma3.4b_results.json"),
    "Gemma 4.e4B": os.path.join(RAW_DIR, "gemma4.e4b_results.json"),
    "Llama 3.1": os.path.join(RAW_DIR, "llama3.1_Results.json"),
    "Mistral": os.path.join(RAW_DIR, "mistral_Results.json"), 
    "Qwen 3.8B": os.path.join(RAW_DIR, "qwen3.8b_Results.json")
}

def load_ground_truth_map(filepath):
    if not os.path.exists(filepath): 
        return {}
    with open(filepath, 'r', encoding='utf-8') as f: 
        gt_list = json.load(f)
    return {entry.get('source_article', '').strip(): entry for entry in gt_list}

def evaluate_single_model(model_name, filepath, gt_map):
    if not os.path.exists(filepath): 
        return None
    with open(filepath, 'r', encoding='utf-8') as f: 
        llm_predictions = json.load(f)

    squad_em_scores = [] 
    f1_scores = []

    for pred_entry in llm_predictions:
        article_text = pred_entry.get('source_article', '').strip()
        if article_text not in gt_map: 
            continue 

        gt_entry = gt_map[article_text]
        gold_spans = gt_entry.get('answer_labels', [])
        pred_spans = pred_entry.get('answer_labels', pred_entry.get('model_spans', [])) 

        g_texts = [g.get('text', '').strip() for g in gold_spans if g.get('text', '').strip()]
        p_texts = [p.get('text', '').strip() for p in pred_spans if p.get('text', '').strip()]

        # SQuAD EM calculation 
        squad_em = EvaluationEngine.evaluate_exact_match(g_texts, p_texts)
        squad_em_scores.append(squad_em)

        # bipartite BERTScore calculation 
        _, _, f1 = EvaluationEngine.evaluate_bipartite_bertscore(g_texts, p_texts)
        f1_scores.append(f1)

    total_evaluated = len(squad_em_scores)
    
    # calculate final means
    mean_em_rate = sum(squad_em_scores) / total_evaluated if total_evaluated > 0 else 0.0
    mean_semantic_f1 = sum(f1_scores) / total_evaluated if total_evaluated > 0 else 0.0

    return {
        "Model": model_name, 
        "Articles Evaluated": total_evaluated, 
        "SQuAD Exact Match": round(mean_em_rate, 4), 
        "DeBERTa-MNLI (Span F1)": round(mean_semantic_f1, 4)
    }

def run_comparative_pipeline():
    print("loading GT...")
    gt_map = load_ground_truth_map(GT_FILE)
    if not gt_map: 
        return
    
    results = []
    
    for model_name, file_path in LLM_FILES.items():
        print(f"processing {model_name}...", end="\r")
        metrics = evaluate_single_model(model_name, file_path, gt_map)
        if metrics: 
            results.append(metrics)
            
    print("Model processing complete!")
    
    df = pd.DataFrame(results).sort_values(by="DeBERTa-MNLI (Span F1)", ascending=False).reset_index(drop=True)
    df.index += 1 

    print("\n" + "="*80)
    print("LLM metrics report:")
    print("="*80)
    print(df.to_string())
    print("="*80 + "\n")

if __name__ == "__main__":
    run_comparative_pipeline()