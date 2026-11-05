# ============================================================================
# ConfliBERT - Conflict & Political Violence NLP Toolkit
# University of Texas at Dallas | Event Data Lab
# ============================================================================

import os
import time
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


# ============================================================================
# PRETRAINED MODEL LOADING
# ============================================================================

# QA models are loaded on first use and kept in an LRU cache.
_qa_model_cache = OrderedDict()
_qa_model_lock = threading.Lock()


def get_qa_model(display_name):
    """Return (model, tokenizer) for a QA_MODELS key, loading it if needed."""
    if display_name not in QA_MODELS:
        display_name = QA_DEFAULT_MODEL
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


def qa_extract_answers(context, question, qa_model, qa_tokenizer):
    """Extract every sufficiently-confident answer span, best first.

    Candidate spans are scored as P(start) * P(end), pooled across all sliding
    windows, then deduplicated by character offset so the overlap between
    adjacent windows cannot yield the same mention twice. Overlapping and
    repeated spans are suppressed, since the runners-up of a strong answer are
    otherwise just off-by-one variants of it.

    Returns an empty list when the model judges the question unanswerable,
    i.e. when the [CLS] score beats the best span by more than QA_NULL_THRESHOLD.
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

    if not candidates:
        return []

    # SQuAD 1.0 models were never trained to abstain, so their [CLS] score is meaningless.
    if qa_model.name_or_path != Salsarra_QA_model_name:
        # Check for ConfliBERT cont-cased-squad2 model
        if qa_model.name_or_path == "shreyasmeher/ConfliBERT-cont-cased-squad2":
            null_threshold = -5.8
        else:
            null_threshold = QA_NULL_THRESHOLD
        if min_null - best_span > null_threshold:
            return []

    candidates.sort(key=lambda c: -c[0])
    cutoff = candidates[0][0] * QA_ANSWER_MIN_RATIO

    answers, spans, seen = [], [], set()
    for score, char_start, char_end in candidates:
        if score < cutoff:
            break
        if any(char_start < e and s < char_end for s, e in spans):
            continue
        text = context[char_start:char_end].strip()
        key = text.lower()
        if not text or key in seen:
            continue
        spans.append((char_start, char_end))
        seen.add(key)
        answers.append({
            'answer': text,
            'score': score,
            'start': char_start,
            'end': char_end,
        })
        if QA_MAX_ANSWERS and len(answers) >= QA_MAX_ANSWERS:
            break
    return answers


# ============================================================================
# CSV BATCH PROCESSING
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

    # Confidence Details
    df['answer_confidence'] = confidences

    # Optional Details, Uncomment to View
    # df['answer_count'] = counts

    df.to_csv(output_path, index=False)

    return output_path


# ============================================================================
# COMMAND LINE
# ============================================================================

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_path = os.path.join(script_dir, 'train.csv')
    
    while True:
        process = input("Select which process to Run (by number) \n 1. ConfliBERT QA (Trained on SQuAD 1.0) \n 2. ConfliBERT scr-uncased (SQuAD 2.0) \n 3. ConfliBERT scr-cased (SQuAD 2.0) \n 4. ConfliBERT cont-uncased (SQuAD 2.0) \n 5. ConfliBERT cont-cased (SQuAD 2.0) \n 6. BERT Base Cased (SQuAD 2.0) \n 7. BERT Base Uncased (SQuAD 2.0) \n 8. Exit \n")
        
        if process == "1":
            output_path = os.path.join(script_dir, 'ConfliBERT_QA-squad1_results.csv')
            process_csv_qa(input_path, output_path, QA_DEFAULT_MODEL)
                
        elif process == "2":
            output_path = os.path.join(script_dir, 'ConfliBERT-scr-uncased-squad2_results.csv')
            process_csv_qa(input_path, output_path, "ConfliBERT scr-uncased (SQuAD 2.0)")

        elif process == "3":
            output_path = os.path.join(script_dir, 'ConfliBERT-scr-cased-squad2_results.csv')
            process_csv_qa(input_path, output_path, "ConfliBERT scr-cased (SQuAD 2.0)")

        elif process == "4":
            output_path = os.path.join(script_dir, 'ConfliBERT-cont-uncased-squad2_results.csv')
            process_csv_qa(input_path, output_path, "ConfliBERT cont-uncased (SQuAD 2.0)")

        elif process == "5":
            output_path = os.path.join(script_dir, 'ConfliBERT-cont-cased-squad2_results.csv')
            process_csv_qa(input_path, output_path, "ConfliBERT cont-cased (SQuAD 2.0)")

        elif process == "6":
            output_path = os.path.join(script_dir, 'BERT-base-cased-squad2_results.csv')
            process_csv_qa(input_path, output_path, "BERT Base Cased (SQuAD 2.0)")

        elif process == "7":
            output_path = os.path.join(script_dir, 'BERT-base-uncased-squad2_results.csv')
            process_csv_qa(input_path, output_path, "BERT Base Uncased (SQuAD 2.0)")

        elif process == "8":
            print("Exiting...")
            exit()
            
        else:
            print("Invalid Selection")

