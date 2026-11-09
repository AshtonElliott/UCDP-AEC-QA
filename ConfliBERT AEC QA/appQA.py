# ============================================================================
# ConfliBERT - Conflict & Political Violence NLP Toolkit
# University of Texas at Dallas | Event Data Lab
# ============================================================================

import os
import time
import json
import argparse
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import torch
from transformers import AutoTokenizer, AutoModelForQuestionAnswering

# QA model uses TensorFlow (transformers <5) or PyTorch fallback (transformers >=5)
_USE_TF_QA = False
try:
    import tensorflow as tf
    import tf_keras   # noqa: F401
    import keras      # noqa: F401
    from transformers import TFAutoModelForQuestionAnswering
    _USE_TF_QA = True
except (ImportError, ModuleNotFoundError):
    print("Using PyTorch for QA model")

import numpy as np
import pandas as pd
import re
import threading
from collections import OrderedDict
import gc

# ============================================================================
# CONFIGURATION
# ============================================================================

if torch.cuda.is_available():
    device = torch.device('cuda')
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    device = torch.device('mps')
else:
    device = torch.device('cpu')

# ConfliBERT-QA is a BERT-base architecture: 512 learned position embeddings is a
# hard ceiling per forward pass. Longer contexts are handled by sliding overlapping
# QA_WINDOW-sized windows across the context and keeping the best-scoring span.
QA_WINDOW = 512
QA_STRIDE = 128
QA_MAX_ANSWER_LENGTH = 50

# Multi-answer extraction. The model assigns a probability to every possible
# span, so it has no notion of "finding" a fixed number of answers: the length
# of the returned list is governed by QA_ANSWER_MIN_RATIO and by overlap
# suppression, not by a top-k cutoff. Lower the ratio to retrieve more.
QA_ANSWER_MIN_RATIO = 0.10      # keep answers scoring >= this fraction of the best
QA_MAX_ANSWERS = 0              # 0 = unlimited; positive values cap the list
QA_CANDIDATES_PER_WINDOW = 200  # spans pulled from each window before merging
QA_CSV_SEPARATOR = ' | '

# SQuAD 2.0 abstention. The model signals "no answer" by scoring the [CLS] token
# highly. A question is abstained on when the [CLS] logit sum exceeds the best
# span's logit sum by more than this margin. Raise it to abstain less often,
# lower it (negative values allowed) to abstain more often.
QA_NULL_THRESHOLD = 0.0

# SQuAD 2.0 fine-tunes selectable with --model. All are BERT-base, so the
# QA_WINDOW / abstention settings apply to every one of them.
Salsarra_QA_model_name = "salsarra/ConfliBERT-QA"
QA_MODELS = {
    "ConfliBERT QA (Trained on SQuAD 1.0)": Salsarra_QA_model_name,
    "ConfliBERT scr-cased (SQuAD 2.0)": "shreyasmeher/ConfliBERT-scr-cased-squad2",
    "ConfliBERT scr-uncased (SQuAD 2.0)": "shreyasmeher/ConfliBERT-scr-uncased-squad2",
    "ConfliBERT cont-cased (SQuAD 2.0)": "shreyasmeher/ConfliBERT-cont-cased-squad2",
    "ConfliBERT cont-uncased (SQuAD 2.0)": "shreyasmeher/ConfliBERT-cont-uncased-squad2",
    "BERT Base Cased (SQuAD 2.0)": "shreyasmeher/BERT-base-cased-squad2",
    "BERT Base Uncased (SQuAD 2.0)": "shreyasmeher/BERT-base-uncased-squad2",
}
QA_DEFAULT_MODEL = "ConfliBERT QA (Trained on SQuAD 1.0)"
# Each model is ~420 MB; older ones are evicted beyond this many.
QA_MODEL_CACHE_SIZE = 2

MODEL_KEY_TO_NAME = {
    "1": "ConfliBERT QA (Trained on SQuAD 1.0)",
    "2": "ConfliBERT scr-uncased (SQuAD 2.0)",
    "3": "ConfliBERT scr-cased (SQuAD 2.0)",
    "4": "ConfliBERT cont-uncased (SQuAD 2.0)",
    "5": "ConfliBERT cont-cased (SQuAD 2.0)",
    "6": "BERT Base Cased (SQuAD 2.0)",
    "7": "BERT Base Uncased (SQuAD 2.0)",
}

MODEL_FILE_SLUGS_Q1 = {
    "ConfliBERT QA (Trained on SQuAD 1.0)": "ConfliBERT_QA-squad1_results.json",
    "ConfliBERT scr-uncased (SQuAD 2.0)": "ConfliBERT-scr-uncased-squad2_results.json",
    "ConfliBERT scr-cased (SQuAD 2.0)": "ConfliBERT-scr-cased-squad2_results.json",
    "ConfliBERT cont-uncased (SQuAD 2.0)": "ConfliBERT-cont-uncased-squad2_results.json",
    "ConfliBERT cont-cased (SQuAD 2.0)": "ConfliBERT-cont-cased-squad2_results.json",
    "BERT Base Cased (SQuAD 2.0)": "BERT-base-cased-squad2_results.json",
    "BERT Base Uncased (SQuAD 2.0)": "BERT-base-uncased-squad2_results.json",
}

# Per-model tuned defaults (calibrated via 5-fold CV to maximize Token F1).
# Overridden if --tau or --alpha flags are passed explicitly on the CLI.
MODEL_TUNED_DEFAULTS_Q1 = {
    "ConfliBERT QA (Trained on SQuAD 1.0)": {"tau": 0.0, "alpha": 0.10},
    "ConfliBERT scr-uncased (SQuAD 2.0)": {"tau": 0.0, "alpha": 0.10},
    "ConfliBERT scr-cased (SQuAD 2.0)": {"tau": 0.0, "alpha": 0.10},
    "ConfliBERT cont-uncased (SQuAD 2.0)": {"tau": 0.0, "alpha": 0.10},
    "ConfliBERT cont-cased (SQuAD 2.0)": {"tau": 0.0, "alpha": 0.10},
    "BERT Base Cased (SQuAD 2.0)": {"tau": 0.0, "alpha": 0.10},
    "BERT Base Uncased (SQuAD 2.0)": {"tau": 0.0, "alpha": 0.10},
}


# ============================================================================
# PRETRAINED MODEL LOADING
# ============================================================================

# QA models are loaded on first use and kept in an LRU cache.
_qa_model_cache = OrderedDict()
_qa_model_lock = threading.Lock()


def get_qa_model(display_name):
    """Return (model, tokenizer) for a QA_MODELS key, loading it if needed."""
    if display_name not in QA_MODELS:
        raise ValueError(
            f"Unknown model: '{display_name}'. Valid options are:\n  "
            + "\n  ".join(f"- {k}" for k in QA_MODELS.keys())
        )
    with _qa_model_lock:
        if display_name in _qa_model_cache:
            _qa_model_cache.move_to_end(display_name)
            return _qa_model_cache[display_name]
        model_id = QA_MODELS[display_name]
        # salsarra/ConfliBERT-QA only ships TensorFlow weights.
        if model_id == Salsarra_QA_model_name and _USE_TF_QA:
            model = TFAutoModelForQuestionAnswering.from_pretrained(model_id)
        elif model_id == Salsarra_QA_model_name:
            model = AutoModelForQuestionAnswering.from_pretrained(model_id, from_tf=True).to(device)
            model.eval()
        else:
            model = AutoModelForQuestionAnswering.from_pretrained(model_id).to(device)
            model.eval()
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        _qa_model_cache[display_name] = (model, tokenizer)
        while len(_qa_model_cache) > QA_MODEL_CACHE_SIZE:
            _qa_model_cache.popitem(last=False)
            gc.collect()
            if device.type == 'cuda':
                torch.cuda.empty_cache()
        return model, tokenizer


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def handle_error(e):
    msg = str(e)
    match = re.search(
        r"The size of tensor a \((\d+)\) must match the size of tensor b \((\d+)\)", msg
    )
    if match:
        return f"Error: Input ({match.group(1)} tokens) exceeds model limit ({match.group(2)})"
    match_qa = re.search(r"indices\[0,(\d+)\] = \d+ is not in \[0, (\d+)\)", msg)
    if match_qa:
        return f"Error: Input too long for model (limit: {match_qa.group(2)} tokens)"
    return f"Error: {msg}"


# ============================================================================
# INFERENCE FUNCTIONS
# ============================================================================

def _qa_context_mask(encoding, index, input_ids, attention_mask, tokenizer):
    """Boolean mask of positions that belong to the context, not the question."""
    try:
        seq_ids = encoding.sequence_ids(index)
    except (AttributeError, ValueError):
        seq_ids = None
    if seq_ids is not None:
        return np.array([s == 1 for s in seq_ids], dtype=bool)
    # Slow tokenizers expose no sequence_ids; token_type_ids marks the context as 1.
    token_type_ids = encoding.get('token_type_ids')
    if token_type_ids is None:
        return attention_mask[index].astype(bool)
    types = np.asarray(token_type_ids[index])
    mask = (types == 1) & attention_mask[index].astype(bool)
    mask &= input_ids[index] != tokenizer.sep_token_id
    return mask


def _masked_softmax(logits, mask):
    """Softmax restricted to mask; positions outside it receive probability 0."""
    scores = np.where(mask, logits, -np.inf)
    scores = scores - scores.max()
    probs = np.exp(scores)
    total = probs.sum()
    return probs / total if total > 0 else probs


def qa_extract_answers(context, question, qa_model, qa_tokenizer,
                       tau=QA_NULL_THRESHOLD, alpha=QA_ANSWER_MIN_RATIO,
                       return_raw=False):
    """Extract sufficiently-confident answer spans, best first.

    If return_raw=True, returns (answers, null_diff, raw_candidates).
    """
    is_tf = not isinstance(qa_model, torch.nn.Module)
    # 'only_second' keeps the question intact and windows the context alone.
    inputs = qa_tokenizer(
        question, context,
        return_tensors='tf' if is_tf else 'pt',
        truncation='only_second',
        max_length=QA_WINDOW,
        stride=QA_STRIDE,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding='max_length',
    )
    model_inputs = {
        k: v for k, v in inputs.items()
        if k in ('input_ids', 'attention_mask', 'token_type_ids')
    }

    if is_tf:
        outputs = qa_model(model_inputs)
        start_logits = np.asarray(outputs.start_logits)
        end_logits = np.asarray(outputs.end_logits)
    else:
        model_inputs = {k: v.to(device) for k, v in model_inputs.items()}
        with torch.no_grad():
            outputs = qa_model(**model_inputs)
        start_logits = outputs.start_logits.float().cpu().numpy()
        end_logits = outputs.end_logits.float().cpu().numpy()
    input_ids = np.asarray(inputs['input_ids'])
    attention_mask = np.asarray(inputs['attention_mask'])
    offsets = np.asarray(inputs['offset_mapping'])

    # Restrict spans to end >= start and no longer than QA_MAX_ANSWER_LENGTH.
    width = input_ids.shape[1]
    ones = np.ones((width, width), dtype=bool)
    span_band = np.triu(ones) & np.tril(ones, QA_MAX_ANSWER_LENGTH - 1)

    # Following the SQuAD 2.0 convention, the null score is the lowest [CLS]
    # score over all windows: one window confidently containing the answer is
    # enough, even if the windows that miss it all vote for "no answer".
    min_null, best_span = np.inf, -np.inf
    candidates = []
    for i in range(input_ids.shape[0]):
        mask = _qa_context_mask(inputs, i, input_ids, attention_mask, qa_tokenizer)
        if not mask.any():
            continue
        min_null = min(min_null, float(start_logits[i][0] + end_logits[i][0]))
        valid = span_band & mask[:, None] & mask[None, :]
        pair = start_logits[i][:, None] + end_logits[i][None, :]
        best_span = max(best_span, float(pair[valid].max()))

        starts = _masked_softmax(start_logits[i], mask)
        ends = _masked_softmax(end_logits[i], mask)
        scores = np.where(span_band, starts[:, None] * ends[None, :], 0.0)

        flat = scores.ravel()
        keep = min(QA_CANDIDATES_PER_WINDOW, flat.size)
        for idx in np.argpartition(flat, -keep)[-keep:]:
            score = float(flat[idx])
            if score <= 0.0:
                continue
            start, end = divmod(int(idx), width)
            # Character offsets make spans comparable across overlapping windows.
            char_start, char_end = int(offsets[i][start][0]), int(offsets[i][end][1])
            if char_end <= char_start:
                continue
            candidates.append((score, char_start, char_end))

    null_diff = float(min_null - best_span) if (min_null < np.inf and best_span > -np.inf) else 999.0

    if not candidates:
        if return_raw:
            return [], null_diff, []
        return []

    candidates.sort(key=lambda c: -c[0])
    raw_candidates = []
    raw_spans = []
    for score, char_start, char_end in candidates:
        if any(char_start < e and s < char_end for s, e in raw_spans):
            continue
        text = context[char_start:char_end].strip()
        if not text:
            continue
        raw_spans.append((char_start, char_end))
        raw_candidates.append({
            'answer': text,
            'start': char_start,
            'end': char_end,
            'score': round(float(score), 5),
            'label': 'Answer',
        })

    # Abstention check
    if qa_model.name_or_path != Salsarra_QA_model_name:
        if null_diff > tau:
            answers = []
        else:
            cutoff = raw_candidates[0]['score'] * alpha if raw_candidates else 0.0
            answers = [c for c in raw_candidates if c['score'] >= cutoff]
            if QA_MAX_ANSWERS and len(answers) > QA_MAX_ANSWERS:
                answers = answers[:QA_MAX_ANSWERS]
    else:
        cutoff = raw_candidates[0]['score'] * alpha if raw_candidates else 0.0
        answers = [c for c in raw_candidates if c['score'] >= cutoff]
        if QA_MAX_ANSWERS and len(answers) > QA_MAX_ANSWERS:
            answers = answers[:QA_MAX_ANSWERS]

    if return_raw:
        return answers, null_diff, raw_candidates
    return answers


# ============================================================================
# JSON BATCH PROCESSING
# ============================================================================

def process_json_qa(input_path, output_path=None, model_display_name=QA_DEFAULT_MODEL,
                    tau=QA_NULL_THRESHOLD, alpha=QA_ANSWER_MIN_RATIO,
                    cache_raw=False, cache_output_path=None):
    """Run QA inference on canonical JSON data.

    If cache_raw=True, saves raw candidates and null diff to cache_output_path.
    Otherwise, saves production evaluation JSON directly to output_path.
    """
    from tqdm import tqdm
    start = time.perf_counter()
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    qa_model, qa_tokenizer = get_qa_model(model_display_name)

    results = []
    cache_records = []

    for item in tqdm(data, desc=f"QA: {model_display_name}"):
        context = item.get('source_article', item.get('context', ''))
        question = item.get('question', item.get('question1', ''))
        doc_id = item.get('id')

        if not context or not question:
            rec = {k: v for k, v in item.items() if k != 'answer_labels'}
            rec['no_answer'] = "No arms or methods mentioned (Geniune No Answer)"
            results.append(rec)
            if cache_raw:
                cache_records.append({'id': doc_id, 'null_diff': 999.0, 'candidates': []})
            continue

        try:
            found, null_diff, raw_candidates = qa_extract_answers(
                str(context), str(question), qa_model, qa_tokenizer,
                tau=tau, alpha=alpha, return_raw=True,
            )
        except Exception as e:
            rec = {k: v for k, v in item.items() if k != 'answer_labels'}
            rec['error'] = handle_error(e)
            rec['no_answer'] = "No arms or methods mentioned (Geniune No Answer)"
            results.append(rec)
            if cache_raw:
                cache_records.append({'id': doc_id, 'null_diff': 999.0, 'candidates': []})
            continue

        if cache_raw:
            cache_records.append({
                'id': doc_id,
                'null_diff': null_diff,
                'candidates': raw_candidates,
            })
        else:
            rec = {k: v for k, v in item.items() if k != 'answer_labels'}
            if found:
                rec.pop('no_answer', None)
                rec['answer_labels'] = [
                    {
                        'text': a['answer'],
                        'start': a['start'],
                        'end': a['end'],
                        'labels': ['Answer'],
                    }
                    for a in found
                ]
            else:
                rec['no_answer'] = "No arms or methods mentioned (Geniune No Answer)"
                rec.pop('answer_labels', None)
            results.append(rec)

    if cache_raw and cache_output_path:
        os.makedirs(os.path.dirname(os.path.abspath(cache_output_path)), exist_ok=True)
        with open(cache_output_path, 'w', encoding='utf-8') as f:
            json.dump(cache_records, f, ensure_ascii=False, indent=2)
        print(f"Raw cache saved to: {cache_output_path} ({len(cache_records)} records)")
    elif output_path:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"Results saved to: {output_path} ({len(results)} records)")

    runtime = time.perf_counter() - start
    print(f"Model {model_display_name} finished in {runtime:.1f}s ({runtime / 60:.1f} min)")
    return output_path


# ============================================================================
# CSV BATCH PROCESSING (Legacy Backward Compatibility)
# ============================================================================

def process_csv_qa(input_path, output_path, model_display_name=QA_DEFAULT_MODEL):
    from tqdm import tqdm
    start = time.perf_counter()
    df = pd.read_csv(input_path)
    if 'context' not in df.columns or 'question' not in df.columns:
        raise ValueError("CSV must contain 'context' and 'question' columns")
    qa_model, qa_tokenizer = get_qa_model(model_display_name)

    answers, confidences, counts = [], [], []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Answering"):
        if pd.isna(row['context']) or pd.isna(row['question']):
            answers.append("")
            confidences.append("")
            counts.append(0)
            continue
        try:
            found = qa_extract_answers(
                str(row['context']), str(row['question']), qa_model, qa_tokenizer,
            )
        except Exception as e:
            answers.append(handle_error(e))
            confidences.append("")
            counts.append(0)
            continue
        answers.append(QA_CSV_SEPARATOR.join(a['answer'] for a in found))
        confidences.append(
            QA_CSV_SEPARATOR.join(f"{a['score']:.4f}" for a in found)
        )
        counts.append(len(found))

    df['answer'] = answers
    df['label'] = "Answer"
    df['answer_confidence'] = confidences
    df.to_csv(output_path, index=False)
    return output_path


# ============================================================================
# COMMAND LINE
# ============================================================================

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_input = os.path.abspath(os.path.join(script_dir, "..", "evaluation", "data", "groundtruth", "ground_truth_q1.json"))
    default_output_dir = os.path.abspath(os.path.join(script_dir, "..", "evaluation", "data", "ConfliBERT Results"))
    default_cache_dir = os.path.abspath(os.path.join(script_dir, "..", "evaluation", "data", "cache"))

    parser = argparse.ArgumentParser(description="ConfliBERT QA Inference (Question 1: Weapons & Methods)")
    parser.add_argument("--model", type=str, default="all",
                        help="Model to run: 'all' (default), 1-7, or exact model name")
    parser.add_argument("--tau", type=float, default=None,
                        help="Null threshold (default: model-specific tuned default)")
    parser.add_argument("--alpha", type=float, default=None,
                        help="Answer confidence ratio (default: model-specific tuned default)")
    parser.add_argument("--cache-raw", action="store_true",
                        help="Extract and save raw candidate pool to cache (skips standard output)")

    args = parser.parse_args()

    # Determine models to run
    if args.model.lower() == "all":
        models_to_run = list(QA_MODELS.keys())
    elif args.model in MODEL_KEY_TO_NAME:
        models_to_run = [MODEL_KEY_TO_NAME[args.model]]
    elif args.model in QA_MODELS:
        models_to_run = [args.model]
    else:
        # Check comma separated
        keys = [k.strip() for k in args.model.split(",")]
        models_to_run = [MODEL_KEY_TO_NAME.get(k, k) for k in keys if MODEL_KEY_TO_NAME.get(k, k) in QA_MODELS]
        if not models_to_run:
            print(f"Error: Unknown model '{args.model}'. Choose from 1-7, 'all', or: {list(QA_MODELS.keys())}")
            exit(1)

    print(f"Starting Q1 QA inference (Models: {len(models_to_run)}, cache_raw={args.cache_raw})")

    for model_name in models_to_run:
        tuned = MODEL_TUNED_DEFAULTS_Q1.get(model_name, {"tau": QA_NULL_THRESHOLD, "alpha": QA_ANSWER_MIN_RATIO})
        model_tau = args.tau if args.tau is not None else tuned["tau"]
        model_alpha = args.alpha if args.alpha is not None else tuned["alpha"]

        slug = MODEL_FILE_SLUGS_Q1[model_name]
        output_file = os.path.join(default_output_dir, slug)
        cache_base = slug.replace('_results.json', '').replace('.json', '')
        cache_file = os.path.join(default_cache_dir, f"raw_cache_q1_{cache_base}.json")

        print(f"  -> Model '{model_name}': using tau={model_tau}, alpha={model_alpha}")
        process_json_qa(
            input_path=default_input,
            output_path=output_file,
            model_display_name=model_name,
            tau=model_tau,
            alpha=model_alpha,
            cache_raw=args.cache_raw,
            cache_output_path=cache_file,
        )

