import sys
import json
from pathlib import Path
from collections import Counter, defaultdict

BASE_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = BASE_DIR / 'data' / 'evaluation_results'
ARTIFACT_PATH = RESULTS_DIR / "master_evaluation_artifact.json"
EXAMPLE_LIMIT = 3

ERROR_TYPES = (
    "missed_extraction",
    "hallucination",
    "over_extraction",
    "paraphrase",
    "label_crash",
)


def _select_examples(cases, severity_key, limit=EXAMPLE_LIMIT):
    """Pick the most severe examples, preferring one per model when possible."""
    if len(cases) <= limit:
        return cases

    best_per_model = {}
    for case in cases:
        model = case["model"]
        if model not in best_per_model or severity_key(case) < severity_key(best_per_model[model]):
            best_per_model[model] = case

    selected = sorted(best_per_model.values(), key=severity_key)[:limit]
    if len(selected) >= limit:
        return selected

    selected_ids = {id(case) for case in selected}
    for case in sorted(cases, key=severity_key):
        if id(case) in selected_ids:
            continue
        selected.append(case)
        selected_ids.add(id(case))
        if len(selected) >= limit:
            break
    return selected


def _print_counts(cases):
    total = len(cases)
    by_model = Counter(case["model"] for case in cases)
    print(f"* **Total cases:** {total}")
    if by_model:
        model_bits = ", ".join(
            f"{model}: {count}"
            for model, count in sorted(by_model.items(), key=lambda x: (-x[1], x[0]))
        )
        print(f"* **By model:** {model_bits}")
    print()


def _empty_buckets():
    return {error_type: [] for error_type in ERROR_TYPES}


def run_error_analysis(records=None):
    print("\n## Part 4: Error Analysis & SQuAD 2.0 Edge Cases", file=sys.stdout)
    print("> *Automated extraction of specific failure modes across the IE pipeline (Zero-Shot baseline only).*", file=sys.stdout)

    if records is None:
        if not ARTIFACT_PATH.exists():
            return
        with ARTIFACT_PATH.open('r', encoding='utf-8') as f:
            records = json.load(f)

    # Keep Part 4 aligned with plots: analyze zero-shot only (legacy rows with no strategy count as zero-shot).
    records = [
        rec for rec in records
        if rec.get('strategy', 'zero-shot').replace('standard', 'zero-shot') == 'zero-shot'
    ]

    # Collect all matching cases first, then select diverse severe examples.
    candidates = defaultdict(_empty_buckets)

    for rec in records:
        q_num = rec.get('question', 1)
        g_texts = rec.get('g_texts') or []
        p_texts = rec.get('p_texts') or []
        g_text_str = " | ".join(g_texts) if g_texts else "[NO TARGET SPANS EXIST]"
        p_text_str = " | ".join(p_texts) if p_texts else "[NO SPANS PREDICTED]"

        has_ans = rec.get('has_ans', len(g_texts) > 0)
        has_pred = rec.get('has_pred', len(p_texts) > 0)
        set_text_f1 = rec.get('set_text_f1', 0.0)
        bertscore_f1 = rec.get('bertscore_f1', 0.0)
        label_f1 = rec.get('label_f1', 0.0)

        # 1. Missed Extraction (answerable, but model abstained)
        if has_ans and not has_pred:
            candidates[q_num]["missed_extraction"].append({
                "model": rec['model'],
                "gt": g_text_str,
                "spans": len(g_texts),
            })

        # 2. Hallucination (unanswerable, but model predicted spans)
        elif not has_ans and has_pred:
            candidates[q_num]["hallucination"].append({
                "model": rec['model'],
                "pred": p_text_str,
                "spans": rec.get('spans_generated', len(p_texts)),
            })

        # 3. Over-Extraction (answerable, but too many noisy spans)
        elif has_ans and has_pred and rec.get('spans_generated', len(p_texts)) >= 5 and bertscore_f1 < 0.4:
            candidates[q_num]["over_extraction"].append({
                "model": rec['model'],
                "gt": g_text_str,
                "pred": p_text_str,
                "spans": rec.get('spans_generated', len(p_texts)),
                "f1": bertscore_f1,
            })

        # 4. Valid Paraphrasing (no exact span match, high semantic score)
        elif has_ans and has_pred and set_text_f1 == 0.0 and bertscore_f1 > 0.70:
            candidates[q_num]["paraphrase"].append({
                "model": rec['model'],
                "gt": g_text_str,
                "pred": p_text_str,
                "set_text_f1": set_text_f1,
                "f1": bertscore_f1,
            })

        # 5. Label Mismatch (high text match, low labeled-span match)
        elif has_ans and has_pred and set_text_f1 > 0.8 and label_f1 < 0.3:
            g_lbl = [l for l in rec.get('g_labels', []) if l.strip("[]'\" ") != ""]
            p_lbl = [l for l in rec.get('p_labels', []) if l.strip("[]'\" ") != ""]

            g_lbl_clean = " | ".join(g_lbl) if g_lbl else "[NO LABEL]"
            p_lbl_clean = " | ".join(p_lbl) if p_lbl else "[NO LABEL]"

            if g_lbl_clean != p_lbl_clean:
                candidates[q_num]["label_crash"].append({
                    "model": rec['model'],
                    "gt_text": g_text_str,
                    "pred_text": p_text_str,
                    "gt_label": g_lbl_clean,
                    "pred_label": p_lbl_clean,
                    "set_text_f1": set_text_f1,
                    "label_f1": label_f1,
                })

    severity_keys = {
        "missed_extraction": lambda case: (-case["spans"],),
        "hallucination": lambda case: (-case["spans"],),
        "over_extraction": lambda case: (case["f1"], -case["spans"]),
        "paraphrase": lambda case: (-case["f1"],),
        "label_crash": lambda case: (case["label_f1"], -case["set_text_f1"]),
    }
    diagnostics = defaultdict(_empty_buckets)
    for q_num, q_cases in candidates.items():
        for error_type, cases in q_cases.items():
            diagnostics[q_num][error_type] = _select_examples(cases, severity_keys[error_type])

    for q_num in sorted(diagnostics.keys()):
        print(f"\n### Analysis: Question {q_num}")
        q_data = diagnostics[q_num]
        q_all = candidates[q_num]

        if q_all["missed_extraction"]:
            print("\n#### A. Missed Extractions (Failed to extract an existing answer)")
            _print_counts(q_all["missed_extraction"])
            for i, case in enumerate(q_data["missed_extraction"], 1):
                print(f"**Edge Case #{i} ({case['model']})**")
                print(f"* **Ground Truth:** `{case['gt']}`")
                print("* **Model Prediction:** `[ABSTAINED - Predicted Nothing]`")
                print("---\n")

        if q_all["hallucination"]:
            print("\n#### B. Hallucinations (Generated text on an unanswerable article)")
            _print_counts(q_all["hallucination"])
            for i, case in enumerate(q_data["hallucination"], 1):
                print(f"**Edge Case #{i} ({case['model']})**")
                print("* **Ground Truth:** `[EMPTY ARTICLE - Should have abstained]`")
                print(f"* **Model Prediction:** `{case['pred']}`")
                print(f"* **Spans Generated:** {case['spans']}")
                print("---\n")

        if q_all["over_extraction"]:
            print("\n#### C. Over-Extraction (High Verbosity, Low Precision)")
            _print_counts(q_all["over_extraction"])
            for i, case in enumerate(q_data["over_extraction"], 1):
                print(f"**Edge Case #{i} ({case['model']})**")
                print(f"* **Ground Truth:** `{case['gt']}`")
                print(f"* **Model Prediction:** `{case['pred']}`")
                print(
                    f"* **Scores:** Spans Generated = {case['spans']} | "
                    f"Deduped BERTScore = {case['f1']:.4f}"
                )
                print("---\n")

        if q_all["paraphrase"]:
            print("\n#### D. Valid Paraphrasing (High Semantic Match, Zero Exact Match)")
            _print_counts(q_all["paraphrase"])
            for i, case in enumerate(q_data["paraphrase"], 1):
                print(f"**Edge Case #{i} ({case['model']})**")
                print(f"* **Ground Truth:** `{case['gt']}`")
                print(f"* **Model Prediction:** `{case['pred']}`")
                print(
                    f"* **Scores:** Span F1 = {case['set_text_f1']:.4f} | "
                    f"Deduped BERTScore = {case['f1']:.4f}"
                )
                print("---\n")

        if q_all["label_crash"]:
            print("\n#### E. Label Mismatch (High Text Match, Wrong Category)")
            _print_counts(q_all["label_crash"])
            for i, case in enumerate(q_data["label_crash"], 1):
                print(f"**Edge Case #{i} ({case['model']})**")
                print(f"* **Extracted Text:** `{case['gt_text']}` == `{case['pred_text']}`")
                print(f"* **Target Label:** `{case['gt_label']}`")
                print(f"* **Predicted Label:** `{case['pred_label']}`")
                print(
                    f"* **Scores:** Span F1 = {case['set_text_f1']:.4f} | "
                    f"Labeled Span F1 = {case['label_f1']:.4f}"
                )
                print("---\n")
        elif q_num > 1:
            print("\n#### E. Label Mismatch")
            print("*No label mapping errors found for this question.*")


if __name__ == "__main__":
    run_error_analysis()
