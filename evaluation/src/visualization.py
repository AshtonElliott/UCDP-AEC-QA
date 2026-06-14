import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from src.eval_metrics import TokenSpanEvaluator

def generate_evaluation_dashboard(json_file_path):
    # generates a comprehensive 4-panel analytics dashboard for NLP span extraction
    with open(json_file_path, 'r', encoding='utf-8') as f:
        dataset = json.load(f)

    evaluator = TokenSpanEvaluator(iou_threshold=0.5)
    
    # collect data arrays
    doc_ious = []
    global_tp, global_fp, global_fn, global_em = 0, 0, 0, 0

    for entry in dataset:
        tp, fp, fn, doc_iou, em = evaluator.evaluate_document_spans(
            entry.get('answer_labels', []), 
            entry.get('model_spans', []), 
            entry.get('source_article', '')
        )
        doc_ious.append(doc_iou)
        global_tp += tp
        global_fp += fp
        global_fn += fn
        global_em += em

    # match quality buckets
    total_spans = global_tp + global_fn
    partial_matches = global_tp - global_em
    complete_misses = global_fn

    # threshold curve data
    thresholds = [0.1, 0.3, 0.5, 0.7, 0.9, 1.0]
    f1_scores = []
    for t in thresholds:
        temp_eval = TokenSpanEvaluator(iou_threshold=t)
        t_tp, t_fp, t_fn = 0, 0, 0
        for entry in dataset:
            tp, fp, fn, _, _ = temp_eval.evaluate_document_spans(
                entry.get('answer_labels', []), 
                entry.get('model_spans', []), 
                entry.get('source_article', '')
            )
            t_tp += tp
            t_fp += fp
            t_fn += fn
        p = t_tp / (t_tp + t_fp) if (t_tp + t_fp) > 0 else 0
        r = t_tp / (t_tp + t_fn) if (t_tp + t_fn) > 0 else 0
        f1_scores.append((2 * p * r) / (p + r) if (p + r) > 0 else 0)


   # PLOTTING
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f'Evaluation Analytics: {json_file_path}', fontsize=18, fontweight='bold', y=0.98)

    # TOP LEFT: score Distribution
    sns.histplot(doc_ious, bins=10, kde=True, color='royalblue', ax=axes[0, 0])
    axes[0, 0].set_title('Document Token IoU Distribution', fontsize=14)
    axes[0, 0].set_xlabel('IoU Score')
    axes[0, 0].set_ylabel('Number of Documents')

    # TOP RIGHT: threshold robustness
    sns.lineplot(x=thresholds, y=f1_scores, marker='o', color='darkorange', linewidth=2.5, ax=axes[0, 1])
    axes[0, 1].fill_between(thresholds, f1_scores, alpha=0.2, color='darkorange')
    axes[0, 1].set_title('F1 Score vs. Match Strictness', fontsize=14)
    axes[0, 1].set_xlabel('Required IoU Threshold')
    axes[0, 1].set_ylabel('Resulting F1 Score')
    axes[0, 1].set_ylim(0, 1.05)
    axes[0, 1].axvline(0.5, color='gray', linestyle='--', label='Standard (0.5)')
    axes[0, 1].legend()

    # BOTTOM LEFT: match quality (EM vs partial vs miss)
    categories = ['Perfect Match (EM)', 'Partial Match', 'Complete Miss (FN)']
    counts = [global_em, partial_matches, complete_misses]
    colors = ['#2ca02c', '#ff7f0e', '#d62728'] # green, orange, red
    
    sns.barplot(x=categories, y=counts, hue=categories, palette=colors, legend=False, ax=axes[1, 0])
    axes[1, 0].set_title('Extraction Quality Breakdown', fontsize=14)
    axes[1, 0].set_ylabel('Number of Target Spans')
    for i, count in enumerate(counts):
        axes[1, 0].text(i, count + (max(counts)*0.02), str(count), ha='center', fontweight='bold')

    # BOTTOM RIGHT: extraction error matrix (custom)
    matrix_data = np.array([[global_tp, global_fn], 
                            [global_fp, 0]]) # 0 because TN aren't tracked
    
    sns.heatmap(matrix_data, annot=True, fmt='d', cmap='Blues', cbar=False, 
                xticklabels=['Model Found It', 'Model Missed It'], 
                yticklabels=['Valid Human Span', 'Hallucinated Span'], 
                ax=axes[1, 1], annot_kws={"size": 16, "weight": "bold"})
    
    axes[1, 1].set_title('Token Confusion Matrix', fontsize=14)

    plt.tight_layout(rect=[0, 0, 1, 0.96]) 
    plt.show()