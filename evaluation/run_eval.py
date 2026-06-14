import json
import os
from src.eval_metrics import TokenSpanEvaluator

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
EVAL_FILE = os.path.join(BASE_DIR, 'data', 'mistral_Results.json')
IOU_THRESHOLD = 0.5

def execute_evaluation_pipeline():
    if not os.path.exists(EVAL_FILE):
        print(f"Error: Target data matrix missing at {EVAL_FILE}.")
        return

    with open(EVAL_FILE, 'r', encoding='utf-8') as f:
        dataset = json.load(f)

    evaluator = TokenSpanEvaluator(iou_threshold=IOU_THRESHOLD)

    global_tp, global_fp, global_fn = 0, 0, 0
    global_em = 0
    global_token_iou = 0.0
    evaluated_count = 0

    for entry in dataset:
        full_text = entry.get('source_article', '')
        gold_spans = entry.get('answer_labels', [])
        pred_spans = entry.get('model_spans', [])

        tp, fp, fn, doc_iou, em = evaluator.evaluate_document_spans(gold_spans, pred_spans, full_text)

        global_tp += tp
        global_fp += fp
        global_fn += fn
        global_em += em
        global_token_iou += doc_iou
        evaluated_count += 1

    # calculations
    mean_iou = global_token_iou / evaluated_count if evaluated_count > 0 else 0
    precision = global_tp / (global_tp + global_fp) if (global_tp + global_fp) > 0 else 0
    recall = global_tp / (global_tp + global_fn) if (global_tp + global_fn) > 0 else 0
    f1_score = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    # total possible ground truth spans to calculate exact match rate
    total_gold_spans = global_tp + global_fn
    em_rate = global_em / total_gold_spans if total_gold_spans > 0 else (1.0 if global_fp == 0 else 0)

    print("="*60)
    print("PRODUCTION METRICS SUMMARY REPORT")
    print("="*60)
    print(f"Total Unique Articles Audited:   {evaluated_count}")
    print(f"Mean Token-Level IoU Score:      {mean_iou:.4f}")
    print(f"Exact Match (EM) Rate:           {em_rate:.4f} ({global_em}/{total_gold_spans} spans)")
    print(f"Final Precision Metric:          {precision:.4f}")
    print(f"Final Recall Metric:             {recall:.4f}")
    print(f"Final Balanced F1-Score:         {f1_score:.4f}")
    print("="*60)

if __name__ == "__main__":
    execute_evaluation_pipeline()