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
# A window needs room for at least this much context. Below it the overlap would
# consume the whole window, the slide would stop advancing, and the tokenizer
# aborts with a Rust panic that does not inherit from Exception.
QA_MIN_CONTEXT = 128
# Windows scored per forward pass, so a long context cannot put hundreds of
# sequences through the model at once.
QA_MAX_BATCH = 8

# SQuAD 2.0 abstention. The model signals "no answer" by scoring the [CLS] token
# highly. A question is abstained on when the [CLS] logit sum exceeds the best
# span's logit sum by more than this margin. Raise it to abstain less often,
# lower it (negative values allowed) to abstain more often. 
QA_NULL_THRESHOLD = 0.0

# Multi-answer extraction. The model assigns a probability to every possible
# span, so it has no notion of "finding" a fixed number of answers: the length
# of the returned list is governed by QA_ANSWER_MIN_RATIO and by overlap
# suppression, not by a top-k cutoff. Lower the ratio to retrieve more.
QA_ANSWER_MIN_RATIO = 0.00      # keep answers scoring >= this fraction of the best
QA_MAX_ANSWERS = 0              # 0 = unlimited; positive values cap the list
# Spans pulled from each window before deduplication. Overlapping variants of
# the same mention ("the pipeline" / "pipeline") collapse into one, so this has
# to sit well above the number of answers actually expected.
QA_CANDIDATES_PER_WINDOW = 200
# Joins multiple answers into a single CSV cell.
QA_ANSWER_SEPARATOR = ' | '

# SQuAD 2.0 fine-tunes selectable from the menu. All are BERT-base, so the
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

# ----------------------------------------------------------------------------
# Infrastructure labels assigned to QA answers.
#
# The descriptions double as zero-shot hypotheses, so they are phrased as full
# natural-language clauses rather than terse keywords. 'Other' is the fallback
# and is deliberately excluded from the zero-shot candidate set.
# ----------------------------------------------------------------------------
INFRA_LABELS = {
    'Energy': (
        'infrastructure related to energy exploration, production, and distribution'
    ),
    'Water': (
        'infrastructure related to drinking water, purification, irrigation, '
        'wastewater treatment, and sanitation'
    ),
    'Transportation/Marketing': (
        'infrastructure related to the transportation, marketing, and exchange '
        'of commodities'
    ),
    'Energy/Water': (
        'infrastructure that both produces energy and manages water, such as a '
        'dam where hydropower and irrigation functions cannot be separated'
    ),
    'Health': (
        'infrastructure related to public health, including hospitals, clinics, '
        'and ambulances'
    ),
    'Agriculture/Fishing': (
        'infrastructure related to crop cultivation and harvesting, or '
        'infrastructure related to fisheries'
    ),
    'GOVERNMENT/REBEL': (
        'infrastructure operated by a government or an armed group, such as '
        'schools, administrative buildings, courts, police stations, prisons, '
        'checkpoints, and military bases'
    ),
    'Other': 'infrastructure that does not relate to any of the other categories',
}

# High-precision terms only, matched case-insensitively on word boundaries.
# Terms whose label depends on function rather than name -- dam, reservoir,
# canal, bare "well", pumping station, depot, terminal, plant -- are
# deliberately absent so they fall through to the zero-shot stage, which can
# read the surrounding context.
INFRA_LEXICON = {
    'Energy': [
        'pipeline', 'refinery', 'refineries', 'oil well', 'gas well', 'oilfield',
        'oil field', 'gas field', 'drilling rig', 'derrick', 'power plant',
        'power station', 'substation', 'transformer', 'transmission line',
        'power line', 'powerline', 'electricity grid', 'power grid', 'turbine',
        'coal mine', 'solar farm', 'wind farm', 'fuel depot', 'petrol station',
        'gas station', 'filling station', 'kerosene', 'nuclear plant',
        'crude oil', 'generator',
    ],
    'Water': [
        'water treatment', 'wastewater', 'sewage', 'sewer', 'sanitation',
        'latrine', 'borehole', 'water tower', 'water tank', 'water main',
        'water pipe', 'aqueduct', 'desalination', 'purification',
        'drinking water', 'water supply', 'water point', 'irrigation canal',
        'irrigation channel', 'irrigation ditch', 'water pump', 'hand pump',
        'water kiosk', 'septic',
    ],
    'Transportation/Marketing': [
        'bridge', 'highway', 'railway', 'railroad', 'rail line', 'seaport',
        'port', 'harbour', 'harbor', 'airport', 'airstrip', 'runway',
        'bus station', 'bus terminal', 'train station', 'market',
        'marketplace', 'bazaar', 'souk', 'warehouse', 'ferry', 'dock', 'quay',
        'culvert', 'overpass', 'toll', 'truck stop', 'roadway',
    ],
    'Energy/Water': [
        'hydroelectric', 'hydro-electric', 'hydropower', 'multipurpose dam',
    ],
    'Health': [
        'hospital', 'clinic', 'health center', 'health centre', 'health post',
        'dispensary', 'ambulance', 'pharmacy', 'medical center',
        'medical centre', 'infirmary', 'maternity', 'health facility',
        'vaccination',
    ],
    'Agriculture/Fishing': [
        'farmland', 'cropland', 'orchard', 'plantation', 'granary',
        'grain silo', 'grain store', 'livestock', 'cattle', 'poultry',
        'slaughterhouse', 'abattoir', 'fishery', 'fisheries', 'fishing boat',
        'fishing vessel', 'trawler', 'fish pond', 'aquaculture', 'hatchery',
        'greenhouse', 'threshing', 'harvest',
    ],
    'GOVERNMENT/REBEL': [
        'school', 'university', 'college', 'courthouse', 'court',
        'police station', 'prison', 'jail', 'detention center',
        'detention centre', 'military base', 'barracks', 'checkpoint',
        'border post', 'government building', 'ministry', 'municipal building',
        'city hall', 'town hall', 'administrative building', 'post office',
        'tax office', 'customs office', 'polling station', 'army camp',
        'training camp', 'safehouse', 'embassy', 'consulate',
    ],
}
# Zero-shot entailment model used when the lexicon is silent or conflicted.
# Swap for 'facebook/bart-large-mnli' if this checkpoint is unavailable.
INFRA_ZEROSHOT_MODEL = 'MoritzLaurer/deberta-v3-base-zeroshot-v2.0'
INFRA_ZEROSHOT_HYPOTHESIS = 'This text describes {}.'
# Below this top score the zero-shot stage abstains and the answer gets 'Other'.
INFRA_MIN_CONFIDENCE = 0.00
# Tokens of surrounding context handed to the classifier on each side of the span.
INFRA_CONTEXT_RADIUS = 60


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


_infra_lexicon_patterns = {
    label: [re.compile(rf'\b{re.escape(t)}', re.IGNORECASE) for t in terms]
    for label, terms in INFRA_LEXICON.items()
}

# Populated on first use; None means "not tried yet", False means "unavailable".
_infra_zeroshot = None


def _get_infra_zeroshot():
    """Lazily build the zero-shot pipeline so startup stays fast and offline-safe."""
    global _infra_zeroshot
    if _infra_zeroshot is None:
        try:
            from transformers import pipeline
            # The checkpoint is stored in float16, which CPUs without native
            # half-precision support run roughly 10x slower than float32.
            _infra_zeroshot = pipeline(
                'zero-shot-classification',
                model=INFRA_ZEROSHOT_MODEL,
                device=0 if device.type == 'cuda' else -1,
                torch_dtype='auto' if device.type == 'cuda' else torch.float32,
            )
        except Exception:
            _infra_zeroshot = False
    return _infra_zeroshot or None


def _infra_lexicon_hits(text):
    """Labels whose high-precision terms appear in text, with match counts."""
    hits = {}
    for label, patterns in _infra_lexicon_patterns.items():
        count = sum(len(p.findall(text)) for p in patterns)
        if count:
            hits[label] = count
    return hits


def classify_infrastructure(answer, window=None):
    """Assign exactly one INFRA_LABELS key to a QA answer span.

    The lexicon decides only when it is unambiguous; everything else falls
    through to zero-shot entailment over the answer plus surrounding context,
    because labels like Energy/Water are defined by function rather than by the
    name of the structure. Returns (label, confidence, method).
    """
    if not answer or not answer.strip():
        return 'Other', 0.0, 'empty'

    span_hits = _infra_lexicon_hits(answer)
    # Energy and Water terms inside the span itself describe one dual-function
    # structure. Across the wider window they are more likely two separate
    # structures mentioned nearby, so that case is deferred to zero-shot.
    if set(span_hits) == {'Energy', 'Water'}:
        return 'Energy/Water', 1.0, 'lexicon'
    if len(span_hits) == 1:
        return next(iter(span_hits)), 1.0, 'lexicon'

    if not span_hits and window:
        window_hits = _infra_lexicon_hits(window)
        if len(window_hits) == 1:
            return next(iter(window_hits)), 0.75, 'lexicon-context'

    classifier = _get_infra_zeroshot()
    if classifier is None:
        # No model available: fall back to the strongest lexicon signal.
        if span_hits:
            return max(span_hits, key=span_hits.get), 0.5, 'lexicon-tiebreak'
        return 'Other', 0.0, 'unavailable'

    candidates = {v: k for k, v in INFRA_LABELS.items() if k != 'Other'}
    result = classifier(
        window or answer,
        candidate_labels=list(candidates),
        hypothesis_template=INFRA_ZEROSHOT_HYPOTHESIS,
        batch_size=len(candidates),
    )
    label = candidates[result['labels'][0]]
    score = float(result['scores'][0])
    if score < INFRA_MIN_CONFIDENCE:
        return 'Other', score, 'zero-shot-abstain'
    return label, score, 'zero-shot'


def _qa_extract_spans(context, question, qa_model, qa_tokenizer,
                      max_answers=QA_MAX_ANSWERS):
    """Run windowed QA and return every sufficiently-confident span, best first.

    Candidate spans are scored as P(start) * P(end) within their window, which
    makes them comparable across windows, then kept while they score at least
    QA_ANSWER_MIN_RATIO of the best one. The length of the list therefore
    reflects how many distinct mentions the model is confident about rather
    than a fixed top-k.

    Each span is a dict with 'answer', 'window' (the answer plus surrounding
    context for the infrastructure classifier), 'score' (the span probability),
    'logit' (the raw start+end logit sum) and 'confidence' (that logit
    softmaxed against the other spans and the null option). An empty list means
    the model abstained.
    """
    n_special = qa_tokenizer.num_special_tokens_to_add(pair=True)
    n_question = len(qa_tokenizer(question, add_special_tokens=False)['input_ids'])
    room = QA_WINDOW - n_question - n_special
    if room < QA_MIN_CONTEXT:
        raise ValueError(
            f"Question is too long ({n_question} tokens): it leaves only "
            f"{max(room, 0)} of {QA_WINDOW} tokens per window for the context. "
            f"Shorten it to at most {QA_WINDOW - n_special - QA_MIN_CONTEXT} tokens."
        )
    # The overlap has to stay well below the per-window context room, otherwise
    # each window advances by a token or two and the window count explodes.
    stride = min(QA_STRIDE, room // 2)

    is_tf = not isinstance(qa_model, torch.nn.Module)
    # 'only_second' keeps the question intact and windows the context alone.
    inputs = qa_tokenizer(
        question, context,
        return_tensors='tf' if is_tf else 'pt',
        truncation='only_second',
        max_length=QA_WINDOW,
        stride=stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding='longest',
    )
    model_inputs = {
        k: v for k, v in inputs.items()
        if k in ('input_ids', 'attention_mask', 'token_type_ids')
    }

    input_ids = np.asarray(inputs['input_ids'])
    attention_mask = np.asarray(inputs['attention_mask'])
    offsets = np.asarray(inputs['offset_mapping'])
    n_windows, width = input_ids.shape

    # Restrict spans to end >= start and no longer than QA_MAX_ANSWER_LENGTH.
    ones = np.ones((width, width), dtype=bool)
    span_band = np.triu(ones) & np.tril(ones, QA_MAX_ANSWER_LENGTH - 1)

    # Following the SQuAD 2.0 convention, the null score is the lowest [CLS]
    # score over all windows: one window confidently containing the answer is
    # enough, even if the windows that miss it all vote for "no answer".
    min_null, best_logit = np.inf, -np.inf
    candidates = []
    masks = {}
    for batch_start in range(0, n_windows, QA_MAX_BATCH):
        batch_end = min(batch_start + QA_MAX_BATCH, n_windows)
        batch = {k: v[batch_start:batch_end] for k, v in model_inputs.items()}
        if is_tf:
            outputs = qa_model(batch)
            start_logits = np.asarray(outputs.start_logits)
            end_logits = np.asarray(outputs.end_logits)
        else:
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.no_grad():
                outputs = qa_model(**batch)
            start_logits = outputs.start_logits.float().cpu().numpy()
            end_logits = outputs.end_logits.float().cpu().numpy()

        for j in range(batch_end - batch_start):
            i = batch_start + j
            mask = _qa_context_mask(inputs, i, input_ids, attention_mask, qa_tokenizer)
            if not mask.any():
                continue
            masks[i] = mask
            min_null = min(min_null, float(start_logits[j][0] + end_logits[j][0]))

            # Raw logit sums stay around for the abstention test and the
            # reported confidence, both of which compare spans to the null
            # option rather than to each other.
            span_valid = span_band & mask[:, None] & mask[None, :]
            pair = np.where(span_valid, start_logits[j][:, None] + end_logits[j][None, :],
                            -np.inf).ravel()
            best_logit = max(best_logit, float(pair.max()))

            starts = _masked_softmax(start_logits[j], mask)
            ends = _masked_softmax(end_logits[j], mask)
            scores = np.where(span_band, starts[:, None] * ends[None, :], 0.0).ravel()
            k = min(QA_CANDIDATES_PER_WINDOW, scores.size)
            for flat in np.argpartition(scores, -k)[-k:]:
                score = float(scores[flat])
                if score <= 0.0:
                    continue
                start, end = divmod(int(flat), width)
                candidates.append((score, float(pair[flat]), i, start, end))

    if not candidates:
        return []

    # SQuAD 1.0 models were never trained to abstain, so their [CLS] score is meaningless.
    if qa_model.name_or_path != Salsarra_QA_model_name:
        null_threshold = QA_NULL_THRESHOLD

        # Abstention is a single verdict on the question: either the best span in
        # the whole context beats the null option or none of them do.
        if min_null - best_logit > null_threshold:
            return []

    candidates.sort(key=lambda c: -c[0])
    cutoff = candidates[0][0] * QA_ANSWER_MIN_RATIO

    spans = []
    seen = set()
    for score, logit, chunk, start, end in candidates:
        if max_answers and len(spans) >= max_answers:
            break
        # Candidates are sorted, so once a span falls under the cutoff so does
        # everything after it.
        if score < cutoff:
            break

        # Slice the original text rather than decoding tokens: uncased
        # tokenizers would return decoded spans lowercased.
        chunk_offsets = offsets[chunk]
        start_char = int(chunk_offsets[start][0])
        end_char = int(chunk_offsets[end][1])
        answer = context[start_char:end_char].strip()
        if not answer:
            continue
        # Drop near-duplicates: overlapping spans describe one mention, and the
        # stride makes windows re-score the text they share.
        key = re.sub(r'\W+', ' ', answer.lower()).strip()
        if key in seen:
            continue
        if any(start_char < s['end_char'] and s['start_char'] < end_char for s in spans):
            continue
        seen.add(key)

        # Widen to a context window so the classifier can judge function, not
        # just name.
        valid = np.flatnonzero(masks[chunk])
        lo = max(int(valid[0]), start - INFRA_CONTEXT_RADIUS)
        hi = min(int(valid[-1]), end + INFRA_CONTEXT_RADIUS)
        spans.append({
            'answer': answer,
            'window': context[chunk_offsets[lo][0]:chunk_offsets[hi][1]].strip(),
            'score': score,
            'logit': logit,
            'start_char': start_char,
            'end_char': end_char,
        })

    if spans:
        logits = np.array([s['logit'] for s in spans] + [min_null])
        probs = np.exp(logits - logits.max())
        probs /= probs.sum()
        for span, prob in zip(spans, probs):
            span['confidence'] = float(prob)
    return spans


# ============================================================================
# CSV BATCH PROCESSING
# ============================================================================

def process_csv_qa(input_path, output_path, model_display_name=QA_DEFAULT_MODEL,
                   max_answers=QA_MAX_ANSWERS):
    from tqdm import tqdm
    start = time.perf_counter()
    df = pd.read_csv(input_path)
    if 'context' not in df.columns or 'question' not in df.columns:
        raise ValueError("CSV must contain 'context' and 'question' columns")
    qa_model, qa_tokenizer = get_qa_model(model_display_name)
    max_answers = int(max_answers)

    # One row in, one row out: a row answered with several spans joins them
    # into a single cell, positionally aligned across the answer and label
    # columns.
    sep = QA_ANSWER_SEPARATOR
    answers, labels, confidences, methods, counts = [], [], [], [], []
    answer_scores, answer_confs = [], []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Answering"):
        if pd.isna(row['context']) or pd.isna(row['question']):
            answers.append("")
            labels.append("")
            confidences.append("")
            methods.append("")
            answer_scores.append("")
            answer_confs.append("")
            counts.append(0)
            continue
        try:
            spans = _qa_extract_spans(
                str(row['context']), str(row['question']),
                qa_model, qa_tokenizer, max_answers,
            )
        except Exception as e:
            answers.append(handle_error(e))
            labels.append("")
            confidences.append("")
            methods.append("")
            answer_scores.append("")
            answer_confs.append("")
            counts.append(0)
            continue
        if not spans:
            answers.append("")
            labels.append('Other')
            confidences.append("")
            methods.append('no-answer')
            answer_scores.append("")
            answer_confs.append("")
            counts.append(0)
            continue
        classified = [
            classify_infrastructure(span['answer'], span['window']) for span in spans
        ]
        answers.append(sep.join(span['answer'] for span in spans))
        labels.append(sep.join(label for label, _, _ in classified))
        confidences.append(sep.join(f"{score:.4f}" for _, score, _ in classified))
        methods.append(sep.join(method for _, _, method in classified))
        answer_scores.append(sep.join(f"{span['score']:.4f}" for span in spans))
        answer_confs.append(sep.join(f"{span['confidence']:.4f}" for span in spans))
        counts.append(len(spans))

    df['answer'] = answers
    df['label'] = labels

    # Confidence Details
    df['answer_confidence'] = answer_confs
    df['label_confidence'] = confidences

    # Optional Details, Uncomment to View
    #df['n_answers'] = counts
    #df['label_method'] = methods

    df.to_csv(output_path, index=False)

    return output_path


# ============================================================================
# COMMAND LINE
# ============================================================================

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_path = os.path.join(script_dir, 'train2.csv')
    
    while True:
        process = input("Select which process to Run (by number) \n 1. ConfliBERT QA (Trained on SQuAD 1.0) \n 2. ConfliBERT scr-uncased (SQuAD 2.0) \n 3. ConfliBERT scr-cased (SQuAD 2.0) \n 4. ConfliBERT cont-uncased (SQuAD 2.0) \n 5. ConfliBERT cont-cased (SQuAD 2.0) \n 6. BERT Base Cased (SQuAD 2.0) \n 7. BERT Base Uncased (SQuAD 2.0) \n 8. Exit \n")
        
        if process == "8":
            print("Exiting...")
            exit()
        
        QA_ANSWER_MIN_RATIO = input("What is the Alpha value for the answer confidence score? \n")
        QA_NULL_THRESHOLD = input("What is the Tau value for the null score? \n")
        INFRA_MIN_CONFIDENCE = input("What is the Alpha value for the label confidence score? \n")

        if process == "1":
            output_path = os.path.join(script_dir, 'ConfliBERT_QA-squad1_results2.csv')
            process_csv_qa(input_path, output_path, QA_DEFAULT_MODEL)
                
        elif process == "2":
            output_path = os.path.join(script_dir, 'ConfliBERT-scr-uncased-squad2_results2.csv')
            process_csv_qa(input_path, output_path, "ConfliBERT scr-uncased (SQuAD 2.0)")

        elif process == "3":
            output_path = os.path.join(script_dir, 'ConfliBERT-scr-cased-squad2_results2.csv')
            process_csv_qa(input_path, output_path, "ConfliBERT scr-cased (SQuAD 2.0)")

        elif process == "4":
            output_path = os.path.join(script_dir, 'ConfliBERT-cont-uncased-squad2_results2.csv')
            process_csv_qa(input_path, output_path, "ConfliBERT cont-uncased (SQuAD 2.0)")

        elif process == "5":
            output_path = os.path.join(script_dir, 'ConfliBERT-cont-cased-squad2_results2.csv')
            process_csv_qa(input_path, output_path, "ConfliBERT cont-cased (SQuAD 2.0)")

        elif process == "6":
            output_path = os.path.join(script_dir, 'BERT-base-cased-squad2_results2.csv')
            process_csv_qa(input_path, output_path, "BERT Base Cased (SQuAD 2.0)")

        elif process == "7":
            output_path = os.path.join(script_dir, 'BERT-base-uncased-squad2_results2.csv')
            process_csv_qa(input_path, output_path, "BERT Base Uncased (SQuAD 2.0)")

            
        else:
            print("Invalid Selection")
