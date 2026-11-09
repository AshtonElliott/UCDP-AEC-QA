import os
import sys
import json
import glob
import random
import argparse
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evaluation.scripts.core import EvaluationEngine

# STRATIFIED 5-FOLD PARTITIONING
def make_stratified_folds(gold_dict, n_splits=5, seed=42):
    """Partition document IDs into stratified folds based on answerable vs no-answer."""
    ans_ids = [doc_id for doc_id, g in gold_dict.items() if len(g) > 0]
    no_ans_ids = [doc_id for doc_id, g in gold_dict.items() if len(g) == 0]

    rng = random.Random(seed)
    rng.shuffle(ans_ids)
    rng.shuffle(no_ans_ids)

    folds = [[] for _ in range(n_splits)]
    for i, doc_id in enumerate(ans_ids):
        folds[i % n_splits].append(doc_id)
    for i, doc_id in enumerate(no_ans_ids):
        folds[i % n_splits].append(doc_id)

    return folds


# TUNING ENGINE
def extract_predictions_for_doc(cached_doc, tau, alpha):
    """Apply tau and alpha thresholds to cached document candidate pool."""
    null_diff = cached_doc.get('null_diff', 999.0)
    candidates = cached_doc.get('candidates', [])

    if null_diff > tau:
        return []

    if not candidates:
        return []

    top_score = candidates[0].get('score', 0.0)
    cutoff = top_score * alpha
    return [c.get('text', c.get('answer', '')) for c in candidates if c.get('score', 0.0) >= cutoff]


def compute_dataset_metrics(cache_by_id, gold_dict, doc_ids, tau, alpha):
    """Compute mean Span F1 and Token F1 across a subset of documents for given (tau, alpha)."""
    span_scores = []
    token_scores = []
    for doc_id in doc_ids:
        cached = cache_by_id.get(doc_id)
        if not cached:
            continue
        golds = gold_dict.get(doc_id, [])
        preds = extract_predictions_for_doc(cached, tau, alpha)
        _, _, span_f1, token_f1 = EvaluationEngine.evaluate_span_and_token_f1(golds, preds)
        span_scores.append(span_f1)
        token_scores.append(token_f1)

    mean_span = sum(span_scores) / len(span_scores) if span_scores else 0.0
    mean_token = sum(token_scores) / len(token_scores) if token_scores else 0.0
    return mean_span, mean_token


def tune_model_cv(cache_records, gold_dict, model_name, question_label="Q1"):
    """Run 5-Fold Stratified CV tuning for a single cached model balancing both Span & Token F1.

    Uses vectorized/memoized precomputation of per-document scores for ultra-fast grid search.
    """
    cache_by_id = {r['id']: r for r in cache_records if 'id' in r}
    all_doc_ids = [doc_id for doc_id in gold_dict.keys() if doc_id in cache_by_id]

    if not all_doc_ids:
        print(f"Error: No matching document IDs between cache and groundtruth for {model_name}.")
        return None

    subset_gold = {doc_id: gold_dict[doc_id] for doc_id in all_doc_ids}
    folds = make_stratified_folds(subset_gold, n_splits=5, seed=42)

    # 1. Define Search Grids
    all_null_diffs = sorted([
        r['null_diff'] for r in cache_records
        if 'null_diff' in r and -50.0 < r['null_diff'] < 50.0
    ])

    if all_null_diffs:
        # 50 empirical percentiles
        tau_grid = sorted(list(set([
            round(all_null_diffs[int(p * (len(all_null_diffs) - 1) / 50)], 2)
            for p in range(51)
        ] + [0.0, -5.8])))
    else:
        tau_grid = [round(-6.0 + i * 0.25, 2) for i in range(41)]

    # Alpha: 51 steps from 0.00 to 1.00 (step 0.02)
    alpha_grid = [round(i * 0.02, 2) for i in range(51)]

    print(f"\n--- Tuning {question_label}: {model_name} (Objective: SQuAD Token F1) ---")
    print(f"Dataset: {len(all_doc_ids)} docs | Search Grid: {len(tau_grid)} tau x {len(alpha_grid)} alpha = {len(tau_grid)*len(alpha_grid)} pairs")

    # 2. Precompute evaluations per document to make grid search 200x faster
    doc_null_eval = {}
    doc_null_diff = {}
    doc_ans_eval = {doc_id: {} for doc_id in all_doc_ids}

    for doc_id in all_doc_ids:
        golds = gold_dict.get(doc_id, [])
        doc_null_diff[doc_id] = cache_by_id[doc_id].get('null_diff', 999.0)
        # Abstention score (empty predictions)
        _, _, s_f1, t_f1 = EvaluationEngine.evaluate_span_and_token_f1(golds, [])
        doc_null_eval[doc_id] = (s_f1, t_f1)

        candidates = cache_by_id[doc_id].get('candidates', [])
        top_score = candidates[0].get('score', 0.0) if candidates else 0.0

        # Cache evaluations across alpha values (memoizing identical prediction lists)
        seen_preds = {}
        for alpha in alpha_grid:
            cutoff = top_score * alpha
            pred_tuple = tuple(c.get('text', c.get('answer', '')) for c in candidates if c.get('score', 0.0) >= cutoff)
            if pred_tuple not in seen_preds:
                _, _, s_f1, t_f1 = EvaluationEngine.evaluate_span_and_token_f1(golds, list(pred_tuple))
                seen_preds[pred_tuple] = (s_f1, t_f1)
            doc_ans_eval[doc_id][alpha] = seen_preds[pred_tuple]

    def eval_subset(doc_ids, tau, alpha):
        s_tot, t_tot = 0.0, 0.0
        for did in doc_ids:
            if doc_null_diff[did] <= tau:
                s, t = doc_ans_eval[did][alpha]
            else:
                s, t = doc_null_eval[did]
            s_tot += s
            t_tot += t
        n = len(doc_ids)
        return s_tot / n, t_tot / n

    # 3. Baseline & Neutral evaluation
    base_span, base_token = eval_subset(all_doc_ids, tau=-5.8, alpha=0.10)
    neut_span, neut_token = eval_subset(all_doc_ids, tau=0.0, alpha=0.00)

    # 4. 5-Fold Stratified Cross-Validation
    oof_predictions = {}

    for k in range(5):
        test_ids = folds[k]
        train_ids = [doc_id for f in range(5) if f != k for doc_id in folds[f]]

        best_train_token = -1.0
        best_train_span = -1.0
        best_tau, best_alpha = 0.0, 0.0

        for tau in tau_grid:
            for alpha in alpha_grid:
                s_f1, t_f1 = eval_subset(train_ids, tau, alpha)
                if (t_f1 > best_train_token) or (t_f1 == best_train_token and s_f1 > best_train_span):
                    best_train_token = t_f1
                    best_train_span = s_f1
                    best_tau = tau
                    best_alpha = alpha

        # Evaluate optimal parameters on out-of-fold test set
        for doc_id in test_ids:
            oof_predictions[doc_id] = extract_predictions_for_doc(cache_by_id[doc_id], best_tau, best_alpha)

    # Compute overall Out-Of-Fold Span and Token F1
    oof_span_list = []
    oof_token_list = []
    for doc_id in all_doc_ids:
        _, _, s_f1, t_f1 = EvaluationEngine.evaluate_span_and_token_f1(gold_dict[doc_id], oof_predictions[doc_id])
        oof_span_list.append(s_f1)
        oof_token_list.append(t_f1)

    oof_span = sum(oof_span_list) / len(oof_span_list) if oof_span_list else 0.0
    oof_token = sum(oof_token_list) / len(oof_token_list) if oof_token_list else 0.0

    # 5. Global Optimal Parameter
    best_global_token = -1.0
    best_global_span = -1.0
    best_global_tau, best_global_alpha = 0.0, 0.0
    surface = {}
    for tau in tau_grid:
        for alpha in alpha_grid:
            s_f1, t_f1 = eval_subset(all_doc_ids, tau, alpha)
            surface[(tau, alpha)] = {'span': s_f1, 'token': t_f1}
            if (t_f1 > best_global_token) or (t_f1 == best_global_token and s_f1 > best_global_span):
                best_global_token = t_f1
                best_global_span = s_f1
                best_global_tau = tau
                best_global_alpha = alpha

    return {
        'question': question_label,
        'model': model_name,
        'docs': len(all_doc_ids),
        'base_span': round(base_span * 100, 2),
        'base_token': round(base_token * 100, 2),
        'neut_span': round(neut_span * 100, 2),
        'neut_token': round(neut_token * 100, 2),
        'optimal_tau': best_global_tau,
        'optimal_alpha': best_global_alpha,
        'oof_span': round(oof_span * 100, 2),
        'oof_token': round(oof_token * 100, 2),
        'delta_token': round((oof_token - base_token) * 100, 2),
        'delta_span': round((oof_span - base_span) * 100, 2),
        'surface': surface,
    }


# MAIN CLI
def main():
    cache_dir = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "evaluation", "data", "cache"))
    gt_dir = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "evaluation", "data", "groundtruth"))

    gt_q1_path = os.path.join(gt_dir, "ground_truth_q1.json")
    gt_q2_path = os.path.join(gt_dir, "ground_truth_q2.json")

    parser = argparse.ArgumentParser(description="ConfliBERT 5-Fold Stratified Cross-Validation Parameter Tuning")
    parser.add_argument("--q", type=str, default="both", choices=["1", "2", "both"],
                        help="Question to tune: '1', '2', or 'both' (default: both)")
    args = parser.parse_args()

    # Load Ground Truth
    print("Loading Ground Truth datasets...")
    if not os.path.exists(gt_q1_path) or not os.path.exists(gt_q2_path):
        print(f"Error: Ground truth files not found in {gt_dir}")
        exit(1)

    with open(gt_q1_path, 'r', encoding='utf-8') as f:
        gt_q1_raw = json.load(f)
    with open(gt_q2_path, 'r', encoding='utf-8') as f:
        gt_q2_raw = json.load(f)

    gold_q1 = {item['id']: [a['text'] for a in item.get('answer_labels', [])] for item in gt_q1_raw}
    gold_q2 = {item['id']: [a['text'] for a in item.get('answer_labels', [])] for item in gt_q2_raw}

    # Discover Cache Files
    q1_cache_files = glob.glob(os.path.join(cache_dir, "raw_cache_q1_*.json")) if args.q in ["1", "both"] else []
    q2_cache_files = glob.glob(os.path.join(cache_dir, "raw_cache_q2_*.json")) if args.q in ["2", "both"] else []

    if not q1_cache_files and not q2_cache_files:
        print(f"\n[!] No cache files found in: {cache_dir}")
        print("To generate candidate caches, run inference with --cache-raw first:")
        print("    python 'ConfliBERT AEC QA/appQA.py' --cache-raw")
        print("    python 'ConfliBERT AEC QA/appQAQ2.py' --cache-raw")
        exit(0)

    results_q1 = []
    results_q2 = []

    # Tune Question 1 Caches
    for path in sorted(q1_cache_files):
        model_name = os.path.basename(path).replace("raw_cache_q1_", "").replace(".json", "")
        with open(path, 'r', encoding='utf-8') as f:
            cache_data = json.load(f)
        res = tune_model_cv(cache_data, gold_q1, model_name, question_label="Q1")
        if res:
            results_q1.append(res)

    # Tune Question 2 Caches
    for path in sorted(q2_cache_files):
        model_name = os.path.basename(path).replace("raw_cache_q2_", "").replace(".json", "")
        with open(path, 'r', encoding='utf-8') as f:
            cache_data = json.load(f)
        res = tune_model_cv(cache_data, gold_q2, model_name, question_label="Q2")
        if res:
            results_q2.append(res)

    # PRINT RESULTS TABLES 
    print("\n" + "="*105)
    print(" 5-FOLD STRATIFIED CROSS-VALIDATION TUNING RESULTS (SPAN F1 & TOKEN F1)")
    print("="*105)
    header = f"{'Q':<3} {'Model':<33} {'τ*':>6} {'α*':>6} {'Base (Span/Tok)':>18} {'Neut (Span/Tok)':>18} {'OOF (Span/Tok)':>18}"
    print(header)
    print("-" * 105)

    all_results = results_q1 + results_q2
    for r in all_results:
        base_str = f"{r['base_span']:.1f}/{r['base_token']:.1f}%"
        neut_str = f"{r['neut_span']:.1f}/{r['neut_token']:.1f}%"
        oof_str = f"{r['oof_span']:.1f}/{r['oof_token']:.1f}%"
        row = f"{r['question']:<3} {r['model']:<33} {r['optimal_tau']:>6.2f} {r['optimal_alpha']:>6.2f} {base_str:>18} {neut_str:>18} {oof_str:>18}"
        print(row)
    print("="*105)

    # JOINT / UNIVERSAL TUNING
    q1_map = {r['model']: r for r in results_q1}
    q2_map = {r['model']: r for r in results_q2}
    common_models = set(q1_map.keys()) & set(q2_map.keys())

    if common_models:
        print("\n" + "="*95)
        print(" JOINT UNIVERSAL TUNING (Single Compromise Setting for Both Q1 & Q2)")
        print("="*95)
        print(f"{'Model':<35} {'Joint τ*':>10} {'Joint α*':>10} {'Joint Span F1':>16} {'Joint Token F1':>16}")
        print("-" * 95)

        for model in sorted(common_models):
            surf1 = q1_map[model]['surface']
            surf2 = q2_map[model]['surface']
            common_keys = set(surf1.keys()) & set(surf2.keys())

            best_joint_tau, best_joint_alpha = 0.0, 0.0
            best_j_span, best_j_token = -1.0, -1.0

            for (tau, alpha) in common_keys:
                avg_span = (surf1[(tau, alpha)]['span'] + surf2[(tau, alpha)]['span']) / 2.0
                avg_token = (surf1[(tau, alpha)]['token'] + surf2[(tau, alpha)]['token']) / 2.0
                if (avg_token > best_j_token) or (avg_token == best_j_token and avg_span > best_j_span):
                    best_joint_tau = tau
                    best_joint_alpha = alpha
                    best_j_span = avg_span
                    best_j_token = avg_token

            print(f"{model:<35} {best_joint_tau:>10.2f} {best_joint_alpha:>10.2f} {best_j_span*100:>15.2f}% {best_j_token*100:>15.2f}%")
        print("="*95)

    # Save summary CSV with both metrics
    summary_path = os.path.join(cache_dir, "cv_tuning_summary.csv")
    os.makedirs(cache_dir, exist_ok=True)
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write("Question,Model,Optimal_Tau,Optimal_Alpha,Base_Span_F1,Base_Token_F1,Neut_Span_F1,Neut_Token_F1,OOF_Span_F1,OOF_Token_F1,Delta_Token\n")
        for r in all_results:
            f.write(f"{r['question']},{r['model']},{r['optimal_tau']},{r['optimal_alpha']},{r['base_span']},{r['base_token']},{r['neut_span']},{r['neut_token']},{r['oof_span']},{r['oof_token']},{r['delta_token']}\n")
    print(f"\nSummary table saved to: {summary_path}\n")

    if results_q1:
        print("# Recommended Tuned Defaults for appQA.py (Q1):")
        print("MODEL_TUNED_DEFAULTS_Q1 = {")
        for r in results_q1:
            print(f'    "{r["model"]}": {{"tau": {r["optimal_tau"]}, "alpha": {r["optimal_alpha"]}}},')
        print("}\n")

    if results_q2:
        print("# Recommended Tuned Defaults for appQAQ2.py (Q2):")
        print("MODEL_TUNED_DEFAULTS_Q2 = {")
        for r in results_q2:
            print(f'    "{r["model"]}": {{"tau": {r["optimal_tau"]}, "alpha": {r["optimal_alpha"]}}},')
        print("}\n")


if __name__ == "__main__":
    main()
