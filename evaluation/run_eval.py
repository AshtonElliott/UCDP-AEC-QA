import json
import os
import pandas as pd
import warnings
from bert_score import score

# suppress verbose huggingface/transformers warnings 
warnings.filterwarnings("ignore")

from src.eval_metrics import SquadEvaluator

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
GT_FILE = os.path.join(DATA_DIR, 'train.json') 

LLM_FILES = {
    "Gemma 3.4B": os.path.join(DATA_DIR, "gemma3.4b_results.json"),
    "Gemma 4.e4B": os.path.join(DATA_DIR, "gemma4.e4b_results.json"),
    "Llama 3.1": os.path.join(DATA_DIR, "llama3.1_Results.json"),
    "Mistral": os.path.join(DATA_DIR, "mistral_Results (1).json"),
    "Qwen 3.8B": os.path.join(DATA_DIR, "qwen3.8b_Results.json")
}

def load_ground_truth_map(filepath):
    if not os.path.exists(filepath): return {}
    with open(filepath, 'r', encoding='utf-8') as f:
        gt_list = json.load(f)
    return {entry.get('source_article', '').strip(): entry for entry in gt_list}

def evaluate_single_model(model_name, filepath, gt_map, evaluator):
    if not os.path.exists(filepath): return None

    with open(filepath, 'r', encoding='utf-8') as f:
        llm_predictions = json.load(f)

    global_em_hits = 0
    global_em_total = 0
    
    cands_for_bert = []
    refs_for_bert = []
    
    true_negative_count = 0 
    false_match_count = 0  

    for pred_entry in llm_predictions:
        article_text = pred_entry.get('source_article', '').strip()
        if article_text not in gt_map: continue 

        gt_entry = gt_map[article_text]
        gold_spans = gt_entry.get('answer_labels', [])
        pred_spans = pred_entry.get('answer_labels', pred_entry.get('model_spans', [])) 

        # structural exact match (SQuAD)
        em_hits, em_total = evaluator.evaluate_exact_match(gold_spans, pred_spans)
        global_em_hits += em_hits
        global_em_total += em_total

        # prepare semantic extraction strings (BERTScore)
        gold_text = " | ".join([g.get('text', '') for g in gold_spans]).strip()
        pred_text = " | ".join([p.get('text', '') for p in pred_spans]).strip()

        if not gold_text and not pred_text:
            true_negative_count += 1
        elif not gold_text or not pred_text:
            # one is empty, the other is not --> semantic score is 0.0
            false_match_count += 1
        else:
            cands_for_bert.append(pred_text)
            refs_for_bert.append(gold_text)

    # execute Batch BERTScore
    if cands_for_bert and refs_for_bert:
        P, R, F1 = score(cands_for_bert, refs_for_bert, lang="en", verbose=False)
        total_f1_sum = F1.sum().item() + true_negative_count 
    else:
        total_f1_sum = true_negative_count

    total_evaluated = len(cands_for_bert) + true_negative_count + false_match_count
    mean_semantic_f1 = total_f1_sum / total_evaluated if total_evaluated > 0 else 0.0
    
    em_rate = global_em_hits / global_em_total if global_em_total > 0 else 0

    return {
        "Model": model_name,
        "Articles Evaluated": total_evaluated,
        "SQuAD Exact Match": round(em_rate, 4),
        "BERTScore (Semantic F1)": round(mean_semantic_f1, 4)
    }

def run_comparative_pipeline():
    print("Loading Ground Truth...")
    gt_map = load_ground_truth_map(GT_FILE)
    if not gt_map: return

    evaluator = SquadEvaluator()
    results = []
    
    print("\nEvaluating Models via BERTScore & SQuAD S.T. (This takes a moment)...")
    for model_name, file_path in LLM_FILES.items():
        metrics = evaluate_single_model(model_name, file_path, gt_map, evaluator)
        if metrics: results.append(metrics)

    df = pd.DataFrame(results).sort_values(by="BERTScore (Semantic F1)", ascending=False).reset_index(drop=True)
    df.index += 1 

    print("\n" + "="*75)
    print("LLM COMPARATIVE METRICS REPORT")
    print("="*75)
    print(df.to_string())
    print("="*75)

if __name__ == "__main__":
    run_comparative_pipeline()