# ============================================================================
# ConfliBERT - Conflict & Political Violence NLP Toolkit
# University of Texas at Dallas | Event Data Lab
# ============================================================================

import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Patch gradio_client bug: bool JSON sub-schemas crash schema parsing
try:
    from gradio_client import utils as _gc_utils
    _original_get_type = _gc_utils.get_type
    def _patched_get_type(schema):
        if isinstance(schema, bool):
            return "Any"
        return _original_get_type(schema)
    _gc_utils.get_type = _patched_get_type

    _original_json_schema = _gc_utils._json_schema_to_python_type
    def _patched_json_schema(schema, defs=None):
        if isinstance(schema, bool):
            return "Any"
        return _original_json_schema(schema, defs)
    _gc_utils._json_schema_to_python_type = _patched_json_schema
except Exception:
    pass

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoModelForTokenClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
    TrainerCallback,
)

# QA model uses TensorFlow (transformers <5) or PyTorch fallback (transformers >=5)
_USE_TF_QA = False
try:
    import tensorflow as tf  # noqa: F401
    import tf_keras   # noqa: F401
    import keras      # noqa: F401
    from transformers import TFAutoModelForQuestionAnswering
    _USE_TF_QA = True
except (ImportError, ModuleNotFoundError):
    from transformers import AutoModelForQuestionAnswering
import gradio as gr
import numpy as np
import pandas as pd
import re
import csv
import tempfile
from sklearn.metrics import (
    accuracy_score as sk_accuracy,
    precision_score as sk_precision,
    recall_score as sk_recall,
    f1_score as sk_f1,
    roc_curve,
    auc as sk_auc,
)
from sklearn.preprocessing import label_binarize
from torch.utils.data import Dataset as TorchDataset
import gc

# LoRA / QLoRA support (optional)
try:
    from peft import LoraConfig, get_peft_model, TaskType
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False

try:
    from transformers import BitsAndBytesConfig
    BNB_AVAILABLE = True
except ImportError:
    BNB_AVAILABLE = False


# ============================================================================
# CONFIGURATION
# ============================================================================

if torch.cuda.is_available():
    device = torch.device('cuda')
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    device = torch.device('mps')
else:
    device = torch.device('cpu')

MAX_TOKEN_LENGTH = 4096

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


def get_system_info():
    """Build an HTML string describing the user's compute environment."""
    import platform
    lines = []

    # Device
    if device.type == 'cuda':
        gpu_name = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        lines.append(f"GPU: {gpu_name} ({vram:.1f} GB VRAM)")
        lines.append("FP16 training: supported")
    elif device.type == 'mps':
        lines.append("GPU: Apple Silicon (MPS)")
        lines.append("FP16 training: not supported on MPS")
    else:
        lines.append("GPU: None detected (using CPU)")
        lines.append("FP16 training: not supported on CPU")

    # CPU / RAM
    import os
    cpu_count = os.cpu_count() or 1
    lines.append(f"CPU cores: {cpu_count}")
    try:
        import psutil
        ram_gb = psutil.virtual_memory().total / (1024 ** 3)
        lines.append(f"RAM: {ram_gb:.1f} GB")
    except ImportError:
        pass

    lines.append(f"Platform: {platform.system()} {platform.machine()}")
    lines.append(f"PyTorch: {torch.__version__}")

    return " · ".join(lines)

FINETUNE_MODELS = {
    "ConfliBERT (recommended for conflict/political text)": "snowood1/ConfliBERT-scr-uncased",
    "BERT Base Uncased": "bert-base-uncased",
    "BERT Base Cased": "bert-base-cased",
    "RoBERTa Base": "roberta-base",
    "ModernBERT Base": "answerdotai/ModernBERT-base",
    "DeBERTa v3 Base": "microsoft/deberta-v3-base",
    "DistilBERT Base Uncased": "distilbert-base-uncased",
}

NER_LABELS = {
    'Organisation': '#3b82f6',
    'Person': '#ef4444',
    'Location': '#10b981',
    'Quantity': '#ff6b35',
    'Weapon': '#8b5cf6',
    'Nationality': '#06b6d4',
    'Temporal': '#ec4899',
    'DocumentReference': '#92400e',
    'MilitaryPlatform': '#f59e0b',
    'Money': '#f472b6',
}

CLASS_NAMES = ['Negative', 'Positive']
MULTI_CLASS_NAMES = ["Armed Assault", "Bombing or Explosion", "Kidnapping", "Other"]

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

INFRA_LABEL_COLORS = {
    'Energy': '#f59e0b',
    'Water': '#06b6d4',
    'Transportation/Marketing': '#8b5cf6',
    'Energy/Water': '#14b8a6',
    'Health': '#ef4444',
    'Agriculture/Fishing': '#10b981',
    'GOVERNMENT/REBEL': '#3b82f6',
    'Other': '#6b7280',
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
INFRA_MIN_CONFIDENCE = 0.40
# Characters of surrounding context handed to the classifier on each side of the span.
INFRA_CONTEXT_RADIUS = 300


# ============================================================================
# PRETRAINED MODEL LOADING
# ============================================================================

qa_model_name = 'salsarra/ConfliBERT-QA'
if _USE_TF_QA:
    qa_model = TFAutoModelForQuestionAnswering.from_pretrained(qa_model_name)
else:
    qa_model = AutoModelForQuestionAnswering.from_pretrained(qa_model_name, from_tf=True)
qa_tokenizer = AutoTokenizer.from_pretrained(qa_model_name)

ner_model_name = 'eventdata-utd/conflibert-named-entity-recognition'
ner_model = AutoModelForTokenClassification.from_pretrained(ner_model_name).to(device)
ner_tokenizer = AutoTokenizer.from_pretrained(ner_model_name)

clf_model_name = 'eventdata-utd/conflibert-binary-classification'
clf_model = AutoModelForSequenceClassification.from_pretrained(clf_model_name).to(device)
clf_tokenizer = AutoTokenizer.from_pretrained(clf_model_name)

multi_clf_model_name = 'eventdata-utd/conflibert-satp-relevant-multilabel'
multi_clf_model = AutoModelForSequenceClassification.from_pretrained(multi_clf_model_name).to(device)
multi_clf_tokenizer = AutoTokenizer.from_pretrained(multi_clf_model_name)


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def get_path(f):
    """Get file path from Gradio file component output."""
    if f is None:
        return None
    return f if isinstance(f, str) else getattr(f, 'name', str(f))


def truncate_text(text, tokenizer, max_length=MAX_TOKEN_LENGTH):
    tokens = tokenizer.encode(text, truncation=False)
    if len(tokens) > max_length:
        tokens = tokens[:max_length - 1] + [tokenizer.sep_token_id]
        return tokenizer.decode(tokens, skip_special_tokens=True)
    return text


def info_callout(text):
    """Wrap markdown text in a styled callout div to avoid Gradio double-border."""
    return (
        "<div class='info-callout-inner' style='"
        "background: #fff7f3; border-left: 3px solid #ff6b35; "
        "padding: 0.75rem 1rem; border-radius: 0 8px 8px 0; "
        "font-size: 0.9rem;'>\n\n"
        f"{text}\n\n</div>"
    )


def handle_error(e, default_limit=512):
    msg = str(e)
    match = re.search(
        r"The size of tensor a \((\d+)\) must match the size of tensor b \((\d+)\)", msg
    )
    if match:
        return (
            f"<span style='color: #ef4444; font-weight: 600;'>"
            f"Error: Input ({match.group(1)} tokens) exceeds model limit ({match.group(2)})</span>"
        )
    match_qa = re.search(r"indices\[0,(\d+)\] = \d+ is not in \[0, (\d+)\)", msg)
    if match_qa:
        return (
            f"<span style='color: #ef4444; font-weight: 600;'>"
            f"Error: Input too long for model (limit: {match_qa.group(2)} tokens)</span>"
        )
    return f"<span style='color: #ef4444; font-weight: 600;'>Error: {msg}</span>"


# ============================================================================
# INFERENCE FUNCTIONS
# ============================================================================

def _qa_context_mask(encoding, index, input_ids, attention_mask):
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
    mask &= input_ids[index] != qa_tokenizer.sep_token_id
    return mask


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
            _infra_zeroshot = pipeline(
                'zero-shot-classification',
                model=INFRA_ZEROSHOT_MODEL,
                device=0 if device.type == 'cuda' else -1,
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
    )
    label = candidates[result['labels'][0]]
    score = float(result['scores'][0])
    if score < INFRA_MIN_CONFIDENCE:
        return 'Other', score, 'zero-shot-abstain'
    return label, score, 'zero-shot'


def _masked_softmax(logits, mask):
    """Softmax restricted to mask; positions outside it receive probability 0."""
    scores = np.where(mask, logits, -np.inf)
    scores = scores - scores.max()
    probs = np.exp(scores)
    total = probs.sum()
    return probs / total if total > 0 else probs


def qa_extract_answers(context, question):
    """Extract every sufficiently-confident answer span, best first.

    Candidate spans are scored as P(start) * P(end), pooled across all sliding
    windows, then deduplicated by character offset so the overlap between
    adjacent windows cannot yield the same mention twice. Overlapping and
    repeated spans are suppressed, since the runners-up of a strong answer are
    otherwise just off-by-one variants of it. Each answer carries a surrounding
    context window, which the infrastructure classifier needs to judge function
    rather than name.
    """
    # 'only_second' keeps the question intact and windows the context alone.
    inputs = qa_tokenizer(
        question, context,
        return_tensors='tf' if _USE_TF_QA else 'pt',
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

    if _USE_TF_QA:
        outputs = qa_model(model_inputs)
    else:
        with torch.no_grad():
            outputs = qa_model(**model_inputs)
    start_logits = np.asarray(outputs.start_logits)
    end_logits = np.asarray(outputs.end_logits)
    input_ids = np.asarray(model_inputs['input_ids'])
    attention_mask = np.asarray(model_inputs['attention_mask'])
    offsets = np.asarray(inputs['offset_mapping'])

    # Restrict spans to end >= start and no longer than QA_MAX_ANSWER_LENGTH.
    width = input_ids.shape[1]
    ones = np.ones((width, width), dtype=bool)
    span_band = np.triu(ones) & np.tril(ones, QA_MAX_ANSWER_LENGTH - 1)

    candidates = []
    for i in range(input_ids.shape[0]):
        mask = _qa_context_mask(inputs, i, input_ids, attention_mask)
        if not mask.any():
            continue
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
            'window': context[
                max(0, char_start - INFRA_CONTEXT_RADIUS):
                char_end + INFRA_CONTEXT_RADIUS
            ],
        })
        if QA_MAX_ANSWERS and len(answers) >= QA_MAX_ANSWERS:
            break
    return answers


def _infra_badge(label):
    color = INFRA_LABEL_COLORS.get(label, '#6b7280')
    return (
        f"<span style='background: {color}; color: #ffffff; font-weight: 600; "
        f"padding: 0.1rem 0.5rem; border-radius: 6px; font-size: 0.8rem;'>"
        f"{label}</span>"
    )


def qa_answers_with_labels(context, question):
    """Extract every answer span and attach an infrastructure label to each."""
    answers = qa_extract_answers(context, question)
    for a in answers:
        label, score, method = classify_infrastructure(a['answer'], a['window'])
        a['label'], a['label_confidence'], a['label_method'] = label, score, method
    return answers


def question_answering(context, question):
    """Return (answer_html, label_html) for the QA tab."""
    if not context or not question:
        return "Please provide both context and question.", ""
    try:
        answers = qa_answers_with_labels(context, question)
        if not answers:
            return (
                "<span style='color: #ef4444; font-weight: 600;'>"
                "No answer found in the provided context.</span>",
                "",
            )
        answer_rows = [
            "<div style='margin-bottom: 0.4rem;'>"
            f"<span style='color: #6b7280; font-size: 0.8rem;'>{n}.</span> "
            f"<span style='color: #10b981; font-weight: 600;'>{a['answer']}</span> "
            f"<span style='color: #6b7280; font-size: 0.8rem;'>"
            f"({a['score']:.1%})</span></div>"
            for n, a in enumerate(answers, 1)
        ]
        label_rows = [
            "<div style='margin-bottom: 0.4rem;'>"
            f"<span style='color: #6b7280; font-size: 0.8rem;'>{n}.</span> "
            f"{_infra_badge(a['label'])} "
            f"<span style='color: #6b7280; font-size: 0.8rem;'>"
            f"{a['label_confidence']:.0%} &middot; {a['label_method']}</span></div>"
            for n, a in enumerate(answers, 1)
        ]
        return (
            "<div>" + "".join(answer_rows) + "</div>",
            "<div>" + "".join(label_rows) + "</div>",
        )
    except Exception as e:
        return handle_error(e), ""


def named_entity_recognition(text, output_format='html'):
    if not text:
        return "Please provide text for analysis."
    try:
        inputs = ner_tokenizer(text, return_tensors='pt', truncation=True)
        with torch.no_grad():
            outputs = ner_model(**inputs)
        results = outputs.logits.argmax(dim=2).squeeze().tolist()
        tokens = ner_tokenizer.convert_ids_to_tokens(inputs['input_ids'].squeeze().tolist())
        tokens = [t.replace('[UNK]', "'") for t in tokens]

        entities = []
        seen_labels = set()
        current_entity = []
        current_label = None

        for i in range(len(tokens)):
            token = tokens[i]
            label = ner_model.config.id2label[results[i]].split('-')[-1]

            if token.startswith('##'):
                if entities:
                    if output_format == 'html':
                        entities[-1][0] += token[2:]
                    elif current_entity:
                        current_entity[-1] = current_entity[-1] + token[2:]
            else:
                if output_format == 'csv':
                    if label != 'O':
                        if label == current_label:
                            current_entity.append(token)
                        else:
                            if current_entity:
                                entities.append([' '.join(current_entity), current_label])
                            current_entity = [token]
                            current_label = label
                    else:
                        if current_entity:
                            entities.append([' '.join(current_entity), current_label])
                            current_entity = []
                            current_label = None
                else:
                    entities.append([token, label])

                if label != 'O':
                    seen_labels.add(label)

        if output_format == 'csv' and current_entity:
            entities.append([' '.join(current_entity), current_label])

        if output_format == 'csv':
            grouped = {}
            for token, label in entities:
                if label != 'O':
                    grouped.setdefault(label, []).append(token)
            parts = []
            for label, toks in grouped.items():
                unique = list(dict.fromkeys(toks))
                parts.append(f"{label}: {' | '.join(unique)}")
            return ' || '.join(parts)

        # HTML output
        highlighted = ""
        for token, label in entities:
            color = NER_LABELS.get(label, 'inherit')
            if label != 'O':
                highlighted += (
                    f"<span style='color: {color}; font-weight: 600;'>{token}</span> "
                )
            else:
                highlighted += f"{token} "

        if seen_labels:
            legend_items = ""
            for label in sorted(seen_labels):
                color = NER_LABELS.get(label, '#666')
                legend_items += (
                    f"<li style='color: {color}; font-weight: 600; "
                    f"background: {color}15; padding: 2px 8px; border-radius: 4px; "
                    f"font-size: 0.85rem;'>{label}</li>"
                )
            legend = (
                f"<div style='margin-top: 1rem; padding-top: 0.75rem; "
                f"border-top: 1px solid #e5e7eb;'>"
                f"<strong>Entities found:</strong>"
                f"<ul style='list-style: none; padding: 0; display: flex; "
                f"flex-wrap: wrap; gap: 0.5rem; margin-top: 0.5rem;'>"
                f"{legend_items}</ul></div>"
            )
            return f"<div style='line-height: 1.8;'>{highlighted}</div>{legend}"
        else:
            return (
                f"<div style='line-height: 1.8;'>{highlighted}</div>"
                f"<div style='color: #888; margin-top: 0.5rem;'>No entities detected.</div>"
            )

    except Exception as e:
        return handle_error(e)


def predict_with_model(text, model, tokenizer):
    """Run inference with an arbitrary classification model."""
    model.eval()
    dev = next(model.parameters()).device
    inputs = tokenizer(
        text, return_tensors='pt', truncation=True, padding=True, max_length=512
    )
    inputs = {k: v.to(dev) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    probs = torch.softmax(outputs.logits, dim=1).squeeze()
    predicted = torch.argmax(probs).item()
    num_classes = probs.shape[0] if probs.dim() > 0 else 1

    lines = []
    for i in range(num_classes):
        p = probs[i].item() * 100 if probs.dim() > 0 else probs.item() * 100
        if i == predicted:
            lines.append(
                f"<span style='color: #10b981; font-weight: 600;'>"
                f"Class {i}: {p:.2f}% (predicted)</span>"
            )
        else:
            lines.append(f"<span style='color: #9ca3af;'>Class {i}: {p:.2f}%</span>")
    return "<br>".join(lines)


def text_classification(text, custom_model=None, custom_tokenizer=None):
    if not text:
        return "Please provide text for classification."
    try:
        # Use custom model if loaded
        if custom_model is not None and custom_tokenizer is not None:
            return predict_with_model(text, custom_model, custom_tokenizer)

        # Pretrained binary classifier
        inputs = clf_tokenizer(
            text, return_tensors='pt', truncation=True, padding=True
        ).to(device)
        with torch.no_grad():
            outputs = clf_model(**inputs)
        predicted = torch.argmax(outputs.logits, dim=1).item()
        confidence = torch.softmax(outputs.logits, dim=1).max().item() * 100

        if predicted == 1:
            return (
                f"<span style='color: #10b981; font-weight: 600;'>"
                f"Positive -- Related to conflict, violence, or politics. "
                f"(Confidence: {confidence:.1f}%)</span>"
            )
        else:
            return (
                f"<span style='color: #ef4444; font-weight: 600;'>"
                f"Negative -- Not related to conflict, violence, or politics. "
                f"(Confidence: {confidence:.1f}%)</span>"
            )
    except Exception as e:
        return handle_error(e)


def multilabel_classification(text, custom_model=None, custom_tokenizer=None):
    if not text:
        return "Please provide text for classification."
    try:
        # Use custom model if loaded
        if custom_model is not None and custom_tokenizer is not None:
            return predict_with_model(text, custom_model, custom_tokenizer)

        inputs = multi_clf_tokenizer(
            text, return_tensors='pt', truncation=True, padding=True
        ).to(device)
        with torch.no_grad():
            outputs = multi_clf_model(**inputs)
        probs = torch.sigmoid(outputs.logits).squeeze().tolist()

        results = []
        for i in range(len(probs)):
            conf = probs[i] * 100
            if probs[i] >= 0.5:
                results.append(
                    f"<span style='color: #10b981; font-weight: 600;'>"
                    f"{MULTI_CLASS_NAMES[i]}: {conf:.1f}%</span>"
                )
            else:
                results.append(
                    f"<span style='color: #9ca3af;'>"
                    f"{MULTI_CLASS_NAMES[i]}: {conf:.1f}%</span>"
                )
        return "<br>".join(results)
    except Exception as e:
        return handle_error(e)


# ============================================================================
# CSV BATCH PROCESSING
# ============================================================================

def process_csv_ner(file):
    path = get_path(file)
    if path is None:
        return None
    df = pd.read_csv(path)
    if 'text' not in df.columns:
        raise ValueError("CSV must contain a 'text' column")

    entities = []
    for text in df['text']:
        if pd.isna(text):
            entities.append("")
        else:
            entities.append(named_entity_recognition(str(text), output_format='csv'))
    df['entities'] = entities

    out = tempfile.NamedTemporaryFile(suffix='_ner_results.csv', delete=False)
    df.to_csv(out.name, index=False)
    return out.name


def process_csv_binary(file, custom_model=None, custom_tokenizer=None):
    path = get_path(file)
    if path is None:
        return None
    df = pd.read_csv(path)
    if 'text' not in df.columns:
        raise ValueError("CSV must contain a 'text' column")

    results = []
    for text in df['text']:
        if pd.isna(text):
            results.append("")
        else:
            html = text_classification(str(text), custom_model, custom_tokenizer)
            results.append(re.sub(r'<[^>]+>', '', html).strip())
    df['classification_results'] = results

    out = tempfile.NamedTemporaryFile(suffix='_classification_results.csv', delete=False)
    df.to_csv(out.name, index=False)
    return out.name


def process_csv_multilabel(file):
    path = get_path(file)
    if path is None:
        return None
    df = pd.read_csv(path)
    if 'text' not in df.columns:
        raise ValueError("CSV must contain a 'text' column")

    results = []
    for text in df['text']:
        if pd.isna(text):
            results.append("")
        else:
            html = multilabel_classification(str(text))
            results.append(re.sub(r'<[^>]+>', '', html).strip())
    df['multilabel_results'] = results

    out = tempfile.NamedTemporaryFile(suffix='_multilabel_results.csv', delete=False)
    df.to_csv(out.name, index=False)
    return out.name


def process_csv_qa(file):
    path = get_path(file)
    if path is None:
        return None
    df = pd.read_csv(path)
    if 'context' not in df.columns or 'question' not in df.columns:
        raise ValueError("CSV must contain 'context' and 'question' columns")

    sep = QA_CSV_SEPARATOR
    answers, scores, labels, confidences, methods, counts = [], [], [], [], [], []
    for _, row in df.iterrows():
        if pd.isna(row['context']) or pd.isna(row['question']):
            found = []
        else:
            try:
                found = qa_answers_with_labels(
                    str(row['context']), str(row['question'])
                )
            except Exception as e:
                answers.append(re.sub(r'<[^>]+>', '', handle_error(e)).strip())
                scores.append("")
                labels.append("")
                confidences.append("")
                methods.append("")
                counts.append(0)
                continue
        answers.append(sep.join(a['answer'] for a in found))
        scores.append(sep.join(f"{a['score']:.4f}" for a in found))
        labels.append(sep.join(a['label'] for a in found))
        confidences.append(sep.join(f"{a['label_confidence']:.4f}" for a in found))
        methods.append(sep.join(a['label_method'] for a in found))
        counts.append(len(found))

    df['answer'] = answers
    df['answer_confidence'] = scores
    df['label'] = labels
    df['label_confidence'] = confidences
    df['label_method'] = methods
    df['answer_count'] = counts

    out = tempfile.NamedTemporaryFile(suffix='_qa_results.csv', delete=False)
    df.to_csv(out.name, index=False)
    return out.name


# ============================================================================
# FINETUNING
# ============================================================================

class TextClassificationDataset(TorchDataset):
    """PyTorch Dataset for text classification with HuggingFace tokenizers."""

    def __init__(self, texts, labels, tokenizer, max_length=512):
        self.encodings = tokenizer(
            texts, truncation=True, padding=True,
            max_length=max_length, return_tensors=None,
        )
        self.labels = labels

    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

    def __len__(self):
        return len(self.labels)


def parse_data_file(file_path):
    """Parse a TSV/CSV data file. Expected format: text<separator>label (no header).
    Labels must be integers. Returns (texts, labels, num_labels)."""
    path = get_path(file_path)
    texts, labels = [], []

    # Detect delimiter from first line
    with open(path, 'r', encoding='utf-8') as f:
        first_line = f.readline()
    delimiter = '\t' if '\t' in first_line else ','

    with open(path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter=delimiter, quotechar='"')
        for row in reader:
            if len(row) < 2:
                continue
            try:
                label = int(row[-1].strip())
                text = row[0].strip() if len(row) == 2 else delimiter.join(row[:-1]).strip()
                if text:
                    texts.append(text)
                    labels.append(label)
            except (ValueError, IndexError):
                continue  # skip header or malformed rows

    if not texts:
        raise ValueError(
            "No valid data rows found. Expected format: text<tab>label (no header row)"
        )

    num_labels = max(labels) + 1
    return texts, labels, num_labels


class LogCallback(TrainerCallback):
    """Captures training logs for display in the UI."""

    def __init__(self):
        self.entries = []

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs:
            self.entries.append({**logs})

    def format(self):
        lines = []
        skip_keys = {
            'total_flos', 'train_runtime', 'train_samples_per_second',
            'train_steps_per_second', 'train_loss',
        }
        for entry in self.entries:
            parts = []
            for k, v in sorted(entry.items()):
                if k in skip_keys:
                    continue
                if isinstance(v, float):
                    parts.append(f"{k}: {v:.4f}")
                elif isinstance(v, (int, np.integer)):
                    parts.append(f"{k}: {v}")
            if parts:
                lines.append("  ".join(parts))
        return "\n".join(lines)


def make_compute_metrics(task_type):
    """Factory for compute_metrics function based on task type."""

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        acc = sk_accuracy(labels, preds)

        if task_type == "Binary":
            return {
                'accuracy': acc,
                'precision': sk_precision(labels, preds, zero_division=0),
                'recall': sk_recall(labels, preds, zero_division=0),
                'f1': sk_f1(labels, preds, zero_division=0),
            }
        else:
            return {
                'accuracy': acc,
                'f1_macro': sk_f1(labels, preds, average='macro', zero_division=0),
                'f1_micro': sk_f1(labels, preds, average='micro', zero_division=0),
                'precision_macro': sk_precision(
                    labels, preds, average='macro', zero_division=0
                ),
                'precision_micro': sk_precision(
                    labels, preds, average='micro', zero_division=0
                ),
                'recall_macro': sk_recall(
                    labels, preds, average='macro', zero_division=0
                ),
                'recall_micro': sk_recall(
                    labels, preds, average='micro', zero_division=0
                ),
            }

    return compute_metrics


def run_finetuning(
    train_file, dev_file, test_file, task_type, model_display_name,
    epochs, batch_size, lr, weight_decay, warmup_ratio, max_seq_len,
    grad_accum, fp16, patience, scheduler,
    use_lora, lora_rank, lora_alpha, use_qlora,
    progress=gr.Progress(track_tqdm=True),
):
    """Main finetuning function. Returns logs, metrics, model state, and visibility updates."""
    try:
        # Validate inputs
        if train_file is None or dev_file is None or test_file is None:
            raise ValueError("Please upload all three data files (train, dev, test).")

        epochs = int(epochs)
        batch_size = int(batch_size)
        max_seq_len = int(max_seq_len)
        grad_accum = int(grad_accum)
        patience = int(patience)

        # Parse data files
        train_texts, train_labels, n_train = parse_data_file(train_file)
        dev_texts, dev_labels, n_dev = parse_data_file(dev_file)
        test_texts, test_labels, n_test = parse_data_file(test_file)

        num_labels = max(n_train, n_dev, n_test)
        if task_type == "Binary" and num_labels > 2:
            raise ValueError(
                f"Binary task selected but found {num_labels} label classes in data. "
                f"Use Multiclass instead."
            )
        if task_type == "Binary":
            num_labels = 2

        # Load model and tokenizer
        model_id = FINETUNE_MODELS[model_display_name]
        tokenizer = AutoTokenizer.from_pretrained(model_id)

        lora_active = False
        if use_qlora:
            if not (PEFT_AVAILABLE and BNB_AVAILABLE and torch.cuda.is_available()):
                raise ValueError(
                    "QLoRA requires a CUDA GPU and the peft + bitsandbytes packages."
                )
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            )
            model = AutoModelForSequenceClassification.from_pretrained(
                model_id, num_labels=num_labels, quantization_config=bnb_config,
            )
        else:
            model = AutoModelForSequenceClassification.from_pretrained(
                model_id, num_labels=num_labels,
            )

        if use_lora or use_qlora:
            if not PEFT_AVAILABLE:
                raise ValueError(
                    "LoRA requires the 'peft' package. Install: pip install peft"
                )
            lora_config = LoraConfig(
                task_type=TaskType.SEQ_CLS,
                r=int(lora_rank),
                lora_alpha=int(lora_alpha),
                lora_dropout=0.1,
                bias="none",
            )
            model.enable_input_require_grads()
            model = get_peft_model(model, lora_config)
            lora_active = True

        # Create datasets
        train_ds = TextClassificationDataset(
            train_texts, train_labels, tokenizer, max_seq_len
        )
        dev_ds = TextClassificationDataset(
            dev_texts, dev_labels, tokenizer, max_seq_len
        )
        test_ds = TextClassificationDataset(
            test_texts, test_labels, tokenizer, max_seq_len
        )

        # Output directory
        output_dir = tempfile.mkdtemp(prefix='conflibert_ft_')

        # Training arguments
        best_metric = 'f1' if task_type == 'Binary' else 'f1_macro'
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size * 2,
            learning_rate=lr,
            weight_decay=weight_decay,
            warmup_ratio=warmup_ratio,
            gradient_accumulation_steps=grad_accum,
            fp16=fp16 and torch.cuda.is_available(),
            eval_strategy='epoch',
            save_strategy='epoch',
            load_best_model_at_end=True,
            metric_for_best_model=best_metric,
            greater_is_better=True,
            logging_steps=10,
            save_total_limit=2,
            lr_scheduler_type=scheduler,
            report_to='none',
            seed=42,
        )

        # Callbacks
        log_callback = LogCallback()
        callbacks = [log_callback]
        if patience > 0:
            callbacks.append(EarlyStoppingCallback(early_stopping_patience=patience))

        # Create Trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_ds,
            eval_dataset=dev_ds,
            compute_metrics=make_compute_metrics(task_type),
            callbacks=callbacks,
        )

        # Train
        train_result = trainer.train()

        # Evaluate on test set
        test_results = trainer.evaluate(test_ds, metric_key_prefix='test')

        # Build log text
        lora_info = ""
        if lora_active:
            method = "QLoRA (4-bit)" if use_qlora else "LoRA"
            lora_info = f"PEFT:  {method}  r={int(lora_rank)}  alpha={int(lora_alpha)}\n"
        header = (
            f"=== Configuration ===\n"
            f"Model: {model_display_name}\n"
            f"       {model_id}\n"
            f"Task:  {task_type} Classification ({num_labels} classes)\n"
            f"Data:  {len(train_texts)} train / {len(dev_texts)} dev / {len(test_texts)} test\n"
            f"Epochs: {epochs}  Batch: {batch_size}  LR: {lr}  Scheduler: {scheduler}\n"
            f"{lora_info}"
            f"\n=== Training Log ===\n"
        )
        runtime = train_result.metrics.get('train_runtime', 0)
        footer = (
            f"\n=== Training Complete ===\n"
            f"Time: {runtime:.1f}s ({runtime / 60:.1f} min)\n"
        )
        log_text = header + log_callback.format() + footer

        # Build metrics DataFrame
        metrics_data = []
        for k, v in sorted(test_results.items()):
            if isinstance(v, (int, float, np.floating, np.integer)) and k != 'test_epoch':
                name = k.replace('test_', '').replace('_', ' ').title()
                metrics_data.append([name, f"{float(v):.4f}"])
        metrics_df = pd.DataFrame(metrics_data, columns=['Metric', 'Score'])

        # Merge LoRA weights back into base model for clean save/inference
        trained_model = trainer.model
        if lora_active and hasattr(trained_model, 'merge_and_unload'):
            trained_model = trained_model.merge_and_unload()
        trained_model = trained_model.cpu()
        trained_model.eval()

        return (
            log_text, metrics_df, trained_model, tokenizer, num_labels,
            gr.Column(visible=True), gr.Column(visible=True),
        )

    except Exception as e:
        error_log = f"Training failed:\n{str(e)}"
        empty_df = pd.DataFrame(columns=['Metric', 'Score'])
        return (
            error_log, empty_df, None, None, None,
            gr.Column(visible=False), gr.Column(visible=False),
        )


# ============================================================================
# MODEL MANAGEMENT (predict, save, load)
# ============================================================================

def predict_finetuned(text, model_state, tokenizer_state, num_labels_state):
    """Run prediction with the finetuned model stored in gr.State."""
    if not text:
        return "Please enter some text."
    if model_state is None:
        return "No model available. Please train a model first."
    return predict_with_model(text, model_state, tokenizer_state)


def save_finetuned_model(model_state, tokenizer_state):
    """Save the finetuned model as a downloadable zip file."""
    if model_state is None:
        return None, "No model to save. Please train a model first."
    try:
        save_dir = tempfile.mkdtemp(prefix='conflibert_save_')
        model_state.save_pretrained(save_dir)
        tokenizer_state.save_pretrained(save_dir)
        import shutil
        zip_path = os.path.join(tempfile.gettempdir(), 'finetuned_model')
        shutil.make_archive(zip_path, 'zip', save_dir)
        return zip_path + '.zip', "Model ready for download."
    except Exception as e:
        return None, f"Error saving model: {str(e)}"


def load_custom_model(path):
    """Load a finetuned classification model from disk."""
    if not path or not os.path.isdir(path):
        return None, None, "Invalid path. Please enter a valid model directory."
    try:
        tokenizer = AutoTokenizer.from_pretrained(path)
        model = AutoModelForSequenceClassification.from_pretrained(path)
        model.eval()
        n = model.config.num_labels
        return model, tokenizer, f"Loaded model with {n} classes from: {path}"
    except Exception as e:
        return None, None, f"Error loading model: {str(e)}"


def reset_custom_model():
    """Reset to the pretrained ConfliBERT binary classifier."""
    return None, None, "Reset to pretrained ConfliBERT binary classifier."


def batch_predict_finetuned(file, model_state, tokenizer_state, num_labels_state):
    """Run batch predictions on a CSV using the finetuned model."""
    if model_state is None:
        return None
    path = get_path(file)
    if path is None:
        return None

    df = pd.read_csv(path)
    if 'text' not in df.columns:
        raise ValueError("CSV must contain a 'text' column")

    model_state.eval()
    dev = next(model_state.parameters()).device

    predictions, confidences = [], []
    for text in df['text']:
        if pd.isna(text):
            predictions.append("")
            confidences.append("")
            continue

        inputs = tokenizer_state(
            str(text), return_tensors='pt', truncation=True,
            padding=True, max_length=512,
        )
        inputs = {k: v.to(dev) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model_state(**inputs)
        probs = torch.softmax(outputs.logits, dim=1).squeeze()
        pred = torch.argmax(probs).item()
        conf = probs[pred].item() * 100
        predictions.append(str(pred))
        confidences.append(f"{conf:.1f}%")

    df['predicted_class'] = predictions
    df['confidence'] = confidences

    out = tempfile.NamedTemporaryFile(suffix='_predictions.csv', delete=False)
    df.to_csv(out.name, index=False)
    return out.name


EXAMPLES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "examples")


def load_example_binary():
    """Load the binary classification example dataset."""
    return (
        os.path.join(EXAMPLES_DIR, "binary", "train.tsv"),
        os.path.join(EXAMPLES_DIR, "binary", "dev.tsv"),
        os.path.join(EXAMPLES_DIR, "binary", "test.tsv"),
        "Binary",
    )


def load_example_multiclass():
    """Load the multiclass classification example dataset."""
    return (
        os.path.join(EXAMPLES_DIR, "multiclass", "train.tsv"),
        os.path.join(EXAMPLES_DIR, "multiclass", "dev.tsv"),
        os.path.join(EXAMPLES_DIR, "multiclass", "test.tsv"),
        "Multiclass",
    )


# ============================================================================
# ACTIVE LEARNING
# ============================================================================

def parse_pool_file(file_path):
    """Parse an unlabeled text pool. Accepts CSV with 'text' column, or one text per line."""
    path = get_path(file_path)
    # Try CSV/TSV with 'text' column first
    try:
        df = pd.read_csv(path)
        if 'text' in df.columns:
            texts = [str(t) for t in df['text'].dropna().tolist()]
            if texts:
                return texts
    except Exception:
        pass
    # Fallback: one text per line
    texts = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                texts.append(line)
    if not texts:
        raise ValueError("No texts found in pool file.")
    return texts


def compute_uncertainty(model, tokenizer, texts, strategy='entropy',
                        max_seq_len=512, batch_size=32):
    """Compute uncertainty scores for unlabeled texts. Higher = more uncertain."""
    model.eval()
    dev = next(model.parameters()).device
    scores = []

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        inputs = tokenizer(
            batch_texts, return_tensors='pt', truncation=True,
            padding=True, max_length=max_seq_len,
        )
        inputs = {k: v.to(dev) for k, v in inputs.items()}
        with torch.no_grad():
            logits = model(**inputs).logits
        probs = torch.softmax(logits, dim=1).cpu().numpy()

        if strategy == 'entropy':
            s = -np.sum(probs * np.log(probs + 1e-10), axis=1)
        elif strategy == 'margin':
            sorted_p = np.sort(probs, axis=1)
            s = -(sorted_p[:, -1] - sorted_p[:, -2])
        else:  # least_confidence
            s = -np.max(probs, axis=1)
        scores.extend(s.tolist())

    return scores


def _build_al_metrics_chart(metrics_history, task_type):
    """Build a Plotly chart of active-learning metrics across rounds."""
    import plotly.graph_objects as go

    if not metrics_history:
        return None

    rounds = [m['round'] for m in metrics_history]
    train_sizes = [m.get('train_size', 0) for m in metrics_history]

    metric_keys = (['f1', 'accuracy', 'precision', 'recall']
                    if task_type == 'Binary'
                    else ['f1_macro', 'accuracy'])

    fig = go.Figure()
    colors = ['#ff6b35', '#3b82f6', '#10b981', '#8b5cf6']

    for i, key in enumerate(metric_keys):
        values = [m.get(key) for m in metrics_history]
        if any(v is not None for v in values):
            fig.add_trace(go.Scatter(
                x=rounds, y=values, mode='lines+markers',
                name=key.replace('_', ' ').title(),
                line=dict(color=colors[i % len(colors)], width=2),
            ))

    fig.add_trace(go.Bar(
        x=rounds, y=train_sizes, name='Train Size',
        marker_color='rgba(200,200,200,0.4)', yaxis='y2',
    ))

    fig.update_layout(
        xaxis_title='Round', yaxis_title='Score', yaxis_range=[0, 1.05],
        yaxis2=dict(title='Train Size', overlaying='y', side='right'),
        template='plotly_white',
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
        height=350, margin=dict(t=40, b=40),
    )
    return fig


def _train_al_model(texts, labels, num_labels, dev_texts, dev_labels,
                    task_type, model_id, epochs, batch_size, lr, max_seq_len,
                    use_lora, lora_rank, lora_alpha):
    """Train a model for one active-learning round. Returns (model, tokenizer, eval_metrics)."""
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_id, num_labels=num_labels,
    )

    if use_lora and PEFT_AVAILABLE:
        lora_cfg = LoraConfig(
            task_type=TaskType.SEQ_CLS,
            r=int(lora_rank), lora_alpha=int(lora_alpha),
            lora_dropout=0.1, bias="none",
        )
        model.enable_input_require_grads()
        model = get_peft_model(model, lora_cfg)

    train_ds = TextClassificationDataset(texts, labels, tokenizer, max_seq_len)
    dev_ds = None
    if dev_texts is not None:
        dev_ds = TextClassificationDataset(dev_texts, dev_labels, tokenizer, max_seq_len)

    output_dir = tempfile.mkdtemp(prefix='conflibert_al_')
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size * 2,
        learning_rate=lr,
        weight_decay=0.01,
        warmup_ratio=0.1,
        eval_strategy='epoch' if dev_ds else 'no',
        save_strategy='no',
        logging_steps=10,
        report_to='none',
        seed=42,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=dev_ds,
        compute_metrics=make_compute_metrics(task_type) if dev_ds else None,
    )
    trainer.train()

    eval_metrics = {}
    if dev_ds:
        results = trainer.evaluate()
        for k, v in results.items():
            if isinstance(v, (int, float, np.floating)):
                eval_metrics[k.replace('eval_', '')] = round(float(v), 4)

    trained_model = trainer.model
    if use_lora and PEFT_AVAILABLE and hasattr(trained_model, 'merge_and_unload'):
        trained_model = trained_model.merge_and_unload()

    return trained_model, tokenizer, eval_metrics


def al_initialize(
    seed_file, pool_file, dev_file, task_type, model_display_name,
    query_strategy, query_size, epochs, batch_size, lr, max_seq_len,
    use_lora, lora_rank, lora_alpha,
    progress=gr.Progress(track_tqdm=True),
):
    """Initialize active learning: train on seed data, query first uncertain batch."""
    try:
        if seed_file is None or pool_file is None:
            raise ValueError("Upload both a labeled seed file and an unlabeled pool file.")

        seed_texts, seed_labels, num_labels = parse_data_file(seed_file)
        pool_texts = parse_pool_file(pool_file)

        dev_texts, dev_labels = None, None
        if dev_file is not None:
            dev_texts, dev_labels, _ = parse_data_file(dev_file)

        if task_type == "Binary":
            num_labels = 2

        query_size = int(query_size)
        model_id = FINETUNE_MODELS[model_display_name]

        trained_model, tokenizer, eval_metrics = _train_al_model(
            seed_texts, seed_labels, num_labels, dev_texts, dev_labels,
            task_type, model_id, int(epochs), int(batch_size), lr,
            int(max_seq_len), use_lora, lora_rank, lora_alpha,
        )

        # Build round-0 metrics
        round_metrics = {'round': 0, 'train_size': len(seed_texts)}
        round_metrics.update(eval_metrics)

        # Query uncertain samples from pool
        scores = compute_uncertainty(
            trained_model, tokenizer, pool_texts, query_strategy, int(max_seq_len),
        )
        top_indices = np.argsort(scores)[-query_size:][::-1].tolist()
        query_texts_batch = [pool_texts[i] for i in top_indices]

        annotation_df = pd.DataFrame({
            'Text': query_texts_batch,
            'Label': [''] * len(query_texts_batch),
        })

        al_state = {
            'labeled_texts': list(seed_texts),
            'labeled_labels': list(seed_labels),
            'pool_texts': pool_texts,
            'pool_available': [i for i in range(len(pool_texts)) if i not in set(top_indices)],
            'current_query_indices': top_indices,
            'dev_texts': dev_texts,
            'dev_labels': dev_labels,
            'num_labels': num_labels,
            'round': 1,
            'metrics_history': [round_metrics],
            'model_id': model_id,
            'model_display_name': model_display_name,
            'task_type': task_type,
            'query_strategy': query_strategy,
            'query_size': query_size,
            'epochs': int(epochs),
            'batch_size': int(batch_size),
            'lr': lr,
            'max_seq_len': int(max_seq_len),
            'use_lora': use_lora,
            'lora_rank': int(lora_rank) if use_lora else 8,
            'lora_alpha': int(lora_alpha) if use_lora else 16,
        }

        trained_model = trained_model.cpu()
        trained_model.eval()

        log_text = (
            f"=== Active Learning Initialized ===\n"
            f"Seed: {len(seed_texts)} labeled  |  Pool: {len(pool_texts)} unlabeled\n"
            f"Model: {model_display_name}\n"
            f"Strategy: {query_strategy}  |  Samples/round: {query_size}\n\n"
            f"--- Round 0 (seed) ---\n"
            f"Train size: {len(seed_texts)}\n"
        )
        for k, v in eval_metrics.items():
            log_text += f"  {k}: {v}\n"
        log_text += (
            f"\n--- Round 1: {len(query_texts_batch)} samples queried ---\n"
            f"Label the samples below, then click 'Submit Labels & Next Round'.\n"
        )

        chart = _build_al_metrics_chart([round_metrics], task_type)

        return (
            al_state, trained_model, tokenizer,
            annotation_df, log_text, chart,
            gr.Column(visible=True),
        )

    except Exception as e:
        return (
            {}, None, None,
            pd.DataFrame(columns=['Text', 'Label']),
            f"Initialization failed:\n{str(e)}",
            None,
            gr.Column(visible=False),
        )


def al_submit_and_continue(
    annotation_df, al_state, al_model, al_tokenizer, prev_log,
    progress=gr.Progress(track_tqdm=True),
):
    """Accept user labels, retrain, query next uncertain batch."""
    try:
        if not al_state or al_model is None:
            raise ValueError("No active session. Initialize first.")

        new_texts = annotation_df['Text'].tolist()
        new_labels = []
        for i, raw in enumerate(annotation_df['Label'].tolist()):
            s = str(raw).strip()
            if s in ('', 'nan'):
                raise ValueError(f"Row {i + 1} has no label. Label all samples first.")
            new_labels.append(int(s))

        num_labels = al_state['num_labels']
        for l in new_labels:
            if l < 0 or l >= num_labels:
                raise ValueError(f"Label {l} out of range [0, {num_labels - 1}].")

        # Add newly labeled samples
        al_state['labeled_texts'].extend(new_texts)
        al_state['labeled_labels'].extend(new_labels)

        queried_set = set(al_state['current_query_indices'])
        al_state['pool_available'] = [
            i for i in al_state['pool_available'] if i not in queried_set
        ]

        current_round = al_state['round']

        # Retrain on all labeled data
        trained_model, tokenizer, eval_metrics = _train_al_model(
            al_state['labeled_texts'], al_state['labeled_labels'],
            num_labels, al_state['dev_texts'], al_state['dev_labels'],
            al_state['task_type'], al_state['model_id'],
            al_state['epochs'], al_state['batch_size'], al_state['lr'],
            al_state['max_seq_len'], al_state['use_lora'],
            al_state['lora_rank'], al_state['lora_alpha'],
        )

        round_metrics = {
            'round': current_round,
            'train_size': len(al_state['labeled_texts']),
        }
        round_metrics.update(eval_metrics)
        al_state['metrics_history'].append(round_metrics)

        # Query next batch from remaining pool
        remaining_pool = al_state['pool_available']
        remaining_texts = [al_state['pool_texts'][i] for i in remaining_pool]

        log_add = (
            f"\n--- Round {current_round} complete ---\n"
            f"Added {len(new_labels)} labels  |  "
            f"Total train: {len(al_state['labeled_texts'])}\n"
        )
        for k, v in eval_metrics.items():
            log_add += f"  {k}: {v}\n"

        if remaining_texts:
            scores = compute_uncertainty(
                trained_model, tokenizer, remaining_texts,
                al_state['query_strategy'], al_state['max_seq_len'],
            )
            q = min(al_state['query_size'], len(remaining_texts))
            top_local = np.argsort(scores)[-q:][::-1].tolist()
            top_pool_indices = [remaining_pool[i] for i in top_local]
            query_texts = [al_state['pool_texts'][i] for i in top_pool_indices]

            al_state['current_query_indices'] = top_pool_indices
            al_state['round'] = current_round + 1

            annotation_out = pd.DataFrame({
                'Text': query_texts,
                'Label': [''] * len(query_texts),
            })
            pool_left = len(remaining_pool) - len(top_pool_indices)
            log_add += (
                f"Pool remaining: {pool_left}\n"
                f"\n--- Round {current_round + 1}: {len(query_texts)} samples queried ---\n"
            )
        else:
            annotation_out = pd.DataFrame(columns=['Text', 'Label'])
            al_state['current_query_indices'] = []
            al_state['round'] = current_round + 1
            log_add += "\nPool exhausted. Active learning complete!\n"

        trained_model = trained_model.cpu()
        trained_model.eval()

        chart = _build_al_metrics_chart(al_state['metrics_history'], al_state['task_type'])
        log_text = prev_log + log_add

        return (
            al_state, trained_model, tokenizer,
            annotation_out, log_text, chart,
        )

    except Exception as e:
        return (
            al_state, al_model, al_tokenizer,
            pd.DataFrame(columns=['Text', 'Label']),
            prev_log + f"\nError: {str(e)}\n",
            None,
        )


def al_save_model(al_model, al_tokenizer):
    """Save the active-learning model as a downloadable zip file."""
    if al_model is None:
        return None, "No model to save. Run at least one round first."
    try:
        save_dir = tempfile.mkdtemp(prefix='conflibert_al_save_')
        al_model.save_pretrained(save_dir)
        al_tokenizer.save_pretrained(save_dir)
        import shutil
        zip_path = os.path.join(tempfile.gettempdir(), 'al_model')
        shutil.make_archive(zip_path, 'zip', save_dir)
        return zip_path + '.zip', "Model ready for download."
    except Exception as e:
        return None, f"Error saving model: {str(e)}"


def load_example_active_learning():
    """Load the active learning example dataset."""
    return (
        os.path.join(EXAMPLES_DIR, "active_learning", "seed.tsv"),
        os.path.join(EXAMPLES_DIR, "active_learning", "pool.txt"),
        os.path.join(EXAMPLES_DIR, "binary", "dev.tsv"),
        "Binary",
    )


def run_comparison(
    train_file, dev_file, test_file, task_type, selected_models,
    epochs, batch_size, lr, cmp_use_lora, cmp_lora_rank, cmp_lora_alpha,
    progress=gr.Progress(track_tqdm=True),
):
    """Train multiple models on the same data and compare performance + ROC curves."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    empty = ("", None, None, None, gr.Column(visible=False))
    try:
        if not selected_models or len(selected_models) < 2:
            return ("Select at least 2 models to compare.",) + empty[1:]
        if train_file is None or dev_file is None or test_file is None:
            return ("Upload all 3 data files first.",) + empty[1:]

        epochs = int(epochs)
        batch_size = int(batch_size)

        train_texts, train_labels, n_train = parse_data_file(train_file)
        dev_texts, dev_labels, n_dev = parse_data_file(dev_file)
        test_texts, test_labels, n_test = parse_data_file(test_file)
        num_labels = max(n_train, n_dev, n_test)
        if task_type == "Binary":
            num_labels = 2

        # Only keep these metrics for the table and bar chart
        if task_type == "Binary":
            keep_metrics = {'Accuracy', 'Precision', 'Recall', 'F1'}
        else:
            keep_metrics = {
                'Accuracy', 'F1 Macro', 'F1 Micro',
                'Precision Macro', 'Recall Macro',
            }

        results = []
        roc_data = {}  # model_name -> (true_labels, probabilities)
        log_lines = []

        for i, model_display_name in enumerate(selected_models):
            model_id = FINETUNE_MODELS[model_display_name]
            short_name = model_display_name.split(" (")[0]
            log_lines.append(f"[{i + 1}/{len(selected_models)}] Training {short_name}...")

            try:
                tokenizer = AutoTokenizer.from_pretrained(model_id)
                model = AutoModelForSequenceClassification.from_pretrained(
                    model_id, num_labels=num_labels,
                )

                cmp_lora_active = False
                if cmp_use_lora and PEFT_AVAILABLE:
                    lora_cfg = LoraConfig(
                        task_type=TaskType.SEQ_CLS,
                        r=int(cmp_lora_rank), lora_alpha=int(cmp_lora_alpha),
                        lora_dropout=0.1, bias="none",
                    )
                    model.enable_input_require_grads()
                    model = get_peft_model(model, lora_cfg)
                    cmp_lora_active = True

                train_ds = TextClassificationDataset(train_texts, train_labels, tokenizer, 512)
                dev_ds = TextClassificationDataset(dev_texts, dev_labels, tokenizer, 512)
                test_ds = TextClassificationDataset(test_texts, test_labels, tokenizer, 512)

                output_dir = tempfile.mkdtemp(prefix='conflibert_cmp_')
                best_metric = 'f1' if task_type == 'Binary' else 'f1_macro'

                training_args = TrainingArguments(
                    output_dir=output_dir,
                    num_train_epochs=epochs,
                    per_device_train_batch_size=batch_size,
                    per_device_eval_batch_size=batch_size * 2,
                    learning_rate=lr,
                    weight_decay=0.01,
                    warmup_ratio=0.1,
                    eval_strategy='epoch',
                    save_strategy='epoch',
                    load_best_model_at_end=True,
                    metric_for_best_model=best_metric,
                    greater_is_better=True,
                    logging_steps=50,
                    save_total_limit=1,
                    report_to='none',
                    seed=42,
                )

                trainer = Trainer(
                    model=model,
                    args=training_args,
                    train_dataset=train_ds,
                    eval_dataset=dev_ds,
                    compute_metrics=make_compute_metrics(task_type),
                )

                train_result = trainer.train()

                # Merge LoRA weights before prediction
                if cmp_lora_active and hasattr(trainer.model, 'merge_and_unload'):
                    trainer.model = trainer.model.merge_and_unload()

                # Get predictions for ROC curves
                pred_output = trainer.predict(test_ds)
                logits = pred_output.predictions
                true_labels = pred_output.label_ids
                probs = torch.softmax(torch.tensor(logits), dim=1).numpy()
                roc_data[short_name] = (true_labels, probs)

                # Collect classification metrics only
                test_results = trainer.evaluate(test_ds, metric_key_prefix='test')
                row = {'Model': short_name}
                for k, v in sorted(test_results.items()):
                    if not isinstance(v, (int, float, np.floating, np.integer)):
                        continue
                    name = k.replace('test_', '').replace('_', ' ').title()
                    if name in keep_metrics:
                        row[name] = round(float(v), 4)
                results.append(row)

                runtime = train_result.metrics.get('train_runtime', 0)
                log_lines.append(f"    Done in {runtime:.1f}s")

                del model, trainer, tokenizer, train_ds, dev_ds, test_ds
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            except Exception as e:
                log_lines.append(f"    Failed: {str(e)}")

        log_lines.append(f"\nComparison complete. {len(results)} models evaluated.")
        log_text = "\n".join(log_lines)

        if not results:
            return log_text, None, None, None, gr.Column(visible=False)

        comparison_df = pd.DataFrame(results)

        # --- Bar chart: classification metrics only ---
        metric_cols = [c for c in comparison_df.columns if c in keep_metrics]
        colors = ['#ff6b35', '#3b82f6', '#10b981', '#8b5cf6', '#f59e0b']
        fig_bar = go.Figure()
        for j, metric in enumerate(metric_cols):
            fig_bar.add_trace(go.Bar(
                name=metric,
                x=comparison_df['Model'],
                y=comparison_df[metric],
                text=comparison_df[metric].apply(
                    lambda x: f'{x:.3f}' if isinstance(x, float) else ''
                ),
                textposition='auto',
                marker_color=colors[j % len(colors)],
            ))
        fig_bar.update_layout(
            barmode='group',
            yaxis_title='Score', yaxis_range=[0, 1.05],
            template='plotly_white',
            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
            height=400, margin=dict(t=40, b=40),
        )

        # --- ROC curves ---
        model_colors = ['#ff6b35', '#3b82f6', '#10b981', '#8b5cf6',
                        '#f59e0b', '#ec4899', '#06b6d4']
        fig_roc = go.Figure()
        for j, (model_name, (labels, probs)) in enumerate(roc_data.items()):
            color = model_colors[j % len(model_colors)]
            if num_labels == 2:
                fpr, tpr, _ = roc_curve(labels, probs[:, 1])
                roc_auc_val = sk_auc(fpr, tpr)
                fig_roc.add_trace(go.Scatter(
                    x=fpr, y=tpr, mode='lines',
                    name=f'{model_name} (AUC = {roc_auc_val:.3f})',
                    line=dict(color=color, width=2),
                ))
            else:
                # Macro-average ROC for multiclass
                labels_bin = label_binarize(labels, classes=list(range(num_labels)))
                all_fpr = np.linspace(0, 1, 200)
                mean_tpr = np.zeros_like(all_fpr)
                for c in range(num_labels):
                    fpr_c, tpr_c, _ = roc_curve(labels_bin[:, c], probs[:, c])
                    mean_tpr += np.interp(all_fpr, fpr_c, tpr_c)
                mean_tpr /= num_labels
                roc_auc_val = sk_auc(all_fpr, mean_tpr)
                fig_roc.add_trace(go.Scatter(
                    x=all_fpr, y=mean_tpr, mode='lines',
                    name=f'{model_name} (macro AUC = {roc_auc_val:.3f})',
                    line=dict(color=color, width=2),
                ))

        # Diagonal reference line
        fig_roc.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1], mode='lines',
            line=dict(dash='dash', color='#ccc', width=1),
            showlegend=False,
        ))
        fig_roc.update_layout(
            xaxis_title='False Positive Rate',
            yaxis_title='True Positive Rate',
            template='plotly_white',
            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
            height=400, margin=dict(t=40, b=40),
        )

        return log_text, comparison_df, fig_bar, fig_roc, gr.Column(visible=True)

    except Exception as e:
        return f"Comparison failed: {str(e)}", None, None, None, gr.Column(visible=False)


# ============================================================================
# THEME & CSS
# ============================================================================

utd_orange = gr.themes.Color(
    c50="#fff7f3", c100="#ffead9", c200="#ffd4b3", c300="#ffb380",
    c400="#ff8c52", c500="#ff6b35", c600="#e8551f", c700="#c2410c",
    c800="#9a3412", c900="#7c2d12", c950="#431407",
)

theme = gr.themes.Soft(
    primary_hue=utd_orange,
    secondary_hue="neutral",
    font=[gr.themes.GoogleFont("Inter"), "ui-sans-serif", "system-ui", "sans-serif"],
)

custom_css = """
/* Top accent bar */
.gradio-container::before {
    content: '';
    display: block;
    height: 4px;
    background: linear-gradient(90deg, #ff6b35, #ff9f40, #ff6b35);
    position: fixed;
    top: 0;
    left: 0;
    right: 0;
    z-index: 1000;
}

/* Active tab styling */
.tab-nav button.selected {
    border-bottom-color: #ff6b35 !important;
    color: #ff6b35 !important;
    font-weight: 600 !important;
}

/* Log output - monospace */
.log-output textarea {
    font-family: 'JetBrains Mono', 'Fira Code', 'Consolas', monospace !important;
    font-size: 0.8rem !important;
    line-height: 1.5 !important;
}

/* Dark mode: info callout adjustment */
.dark .info-callout-inner {
    background: rgba(255, 107, 53, 0.1) !important;
    color: #ffead9 !important;
}

/* Clean container width */
.gradio-container {
    max-width: 1200px !important;
}

/* Smooth transitions */
.gradio-container * {
    transition: background-color 0.2s ease, border-color 0.2s ease !important;
}
"""


# ============================================================================
# GRADIO UI
# ============================================================================

with gr.Blocks(theme=theme, css=custom_css, title="ConfliBERT") as demo:

    # ---- HEADER ----
    gr.Markdown(
        "<div style='text-align: center; padding: 1.5rem 0 0.5rem;'>"
        "<h1 style='font-size: 2.5rem; font-weight: 800; margin: 0;'>"
        "<a href='https://eventdata.utdallas.edu/conflibert/' target='_blank' "
        "style='color: #ff6b35; text-decoration: none;'>ConfliBERT</a></h1>"
        "<p style='color: #888; font-size: 0.95rem; margin: 0.25rem 0 0;'>"
        "A Pretrained Language Model for Conflict and Political Violence</p></div>"
    )

    with gr.Tabs():

        # ================================================================
        # HOME TAB
        # ================================================================
        with gr.Tab("Home"):
            gr.Markdown(
                "## Welcome to ConfliBERT\n\n"
                "ConfliBERT is a pretrained language model built specifically for "
                "conflict and political violence text. This application lets you "
                "run inference with ConfliBERT's pretrained models and fine-tune "
                "your own classifiers on custom data. Use the tabs above to get started."
            )

            with gr.Row(equal_height=True):
                with gr.Column():
                    gr.Markdown(
                        "### Inference\n\n"
                        "Run pretrained ConfliBERT models on your text. "
                        "Each task has its own tab with single-text analysis "
                        "and CSV batch processing.\n\n"
                        "**Named Entity Recognition**\n"
                        "Identify persons, organizations, locations, weapons, "
                        "and other entities in text. Results are color-coded "
                        "by entity type.\n\n"
                        "**Binary Classification**\n"
                        "Determine whether text is related to conflict, violence, "
                        "or politics (positive) or not (negative). You can also "
                        "load a custom fine-tuned model here.\n\n"
                        "**Multilabel Classification**\n"
                        "Score text against four event categories: Armed Assault, "
                        "Bombing/Explosion, Kidnapping, and Other. Each category "
                        "is scored independently.\n\n"
                        "**Question Answering**\n"
                        "Provide a context passage and ask a question. The model "
                        "extracts the most relevant answer span from the text."
                    )
                with gr.Column():
                    gr.Markdown(
                        "### Fine-tuning\n\n"
                        "Train your own binary or multiclass text classifier "
                        "on custom labeled data, all within the browser.\n\n"
                        "**Workflow:**\n"
                        "1. Upload your training, validation, and test data as "
                        "TSV files (or load a built-in example dataset)\n"
                        "2. Pick a base model: ConfliBERT, BERT, RoBERTa, "
                        "ModernBERT, DeBERTa, or DistilBERT\n"
                        "3. Configure training parameters (sensible defaults "
                        "are provided)\n"
                        "4. Train and watch progress in real time\n"
                        "5. Review test-set metrics (accuracy, precision, "
                        "recall, F1)\n"
                        "6. Try your model on new text immediately\n"
                        "7. Run batch predictions on a CSV\n"
                        "8. Save the model and load it later in the "
                        "Classification tab\n\n"
                        "**Advanced features:**\n"
                        "- **LoRA / QLoRA** for parameter-efficient training "
                        "(lower VRAM, faster)\n"
                        "- **Active Learning** tab for iterative labeling "
                        "with uncertainty sampling\n"
                        "- Early stopping with configurable patience\n"
                        "- Learning rate schedulers (linear, cosine, constant)\n"
                        "- Mixed precision training (FP16 on CUDA GPUs)\n"
                        "- Gradient accumulation for larger effective batch sizes\n"
                        "- Weight decay regularization"
                    )

            gr.Markdown(
                f"---\n\n"
                f"**Your system:** {get_system_info()}"
            )

            gr.Markdown(
                "**Citation:** Brandt, P.T., Alsarra, S., D'Orazio, V., "
                "Heintze, D., Khan, L., Meher, S., Osorio, J. and Sianan, M., "
                "2025. Extractive versus Generative Language Models for Political "
                "Conflict Text Classification. *Political Analysis*, pp.1-29."
            )

        # ================================================================
        # NER TAB
        # ================================================================
        with gr.Tab("Named Entity Recognition"):
            gr.Markdown(info_callout(
                "Identify entities in text such as **persons**, **organizations**, "
                "**locations**, **weapons**, and more. Results are color-coded by type."
            ))
            with gr.Row(equal_height=True):
                with gr.Column():
                    ner_input = gr.Textbox(
                        lines=6,
                        placeholder="Paste or type text to analyze for entities...",
                        label="Input Text",
                    )
                    ner_btn = gr.Button("Analyze Entities", variant="primary")
                with gr.Column():
                    ner_output = gr.HTML(label="Results")

            with gr.Accordion("Batch Processing (CSV)", open=False):
                gr.Markdown(
                    "Upload a CSV file with a `text` column to process "
                    "multiple texts at once."
                )
                with gr.Row():
                    ner_csv_in = gr.File(
                        label="Upload CSV", file_types=[".csv"],
                    )
                    ner_csv_out = gr.File(label="Download Results")
                ner_csv_btn = gr.Button("Process CSV", variant="secondary")

        # ================================================================
        # BINARY CLASSIFICATION TAB
        # ================================================================
        with gr.Tab("Binary Classification"):
            gr.Markdown(info_callout(
                "Classify text as **conflict-related** (positive) or "
                "**not conflict-related** (negative). Uses the pretrained ConfliBERT "
                "binary classifier by default, or load your own finetuned model below."
            ))

            custom_clf_model = gr.State(None)
            custom_clf_tokenizer = gr.State(None)

            with gr.Row(equal_height=True):
                with gr.Column():
                    clf_input = gr.Textbox(
                        lines=6,
                        placeholder="Paste or type text to classify...",
                        label="Input Text",
                    )
                    clf_btn = gr.Button("Classify", variant="primary")
                with gr.Column():
                    clf_output = gr.HTML(label="Results")

            with gr.Accordion("Batch Processing (CSV)", open=False):
                gr.Markdown("Upload a CSV file with a `text` column.")
                with gr.Row():
                    clf_csv_in = gr.File(label="Upload CSV", file_types=[".csv"])
                    clf_csv_out = gr.File(label="Download Results")
                clf_csv_btn = gr.Button("Process CSV", variant="secondary")

            with gr.Accordion("Load Custom Model", open=False):
                gr.Markdown(
                    "Load a finetuned classification model from a local directory "
                    "to use instead of the default pretrained classifier."
                )
                clf_model_path = gr.Textbox(
                    label="Model directory path",
                    placeholder="e.g., ./finetuned_model",
                )
                with gr.Row():
                    clf_load_btn = gr.Button("Load Model", variant="secondary")
                    clf_reset_btn = gr.Button(
                        "Reset to Pretrained", variant="secondary",
                    )
                clf_status = gr.Markdown("")

        # ================================================================
        # MULTILABEL CLASSIFICATION TAB
        # ================================================================
        with gr.Tab("Multilabel Classification"):
            gr.Markdown(info_callout(
                "Identify multiple event types in text. Each category is scored "
                "independently: **Armed Assault**, **Bombing/Explosion**, "
                "**Kidnapping**, **Other**. Categories above 50% confidence "
                "are highlighted. Load a custom finetuned model below."
            ))

            custom_multi_model = gr.State(None)
            custom_multi_tokenizer = gr.State(None)

            with gr.Row(equal_height=True):
                with gr.Column():
                    multi_input = gr.Textbox(
                        lines=6,
                        placeholder="Paste or type text to classify...",
                        label="Input Text",
                    )
                    multi_btn = gr.Button("Classify", variant="primary")
                with gr.Column():
                    multi_output = gr.HTML(label="Results")

            with gr.Accordion("Batch Processing (CSV)", open=False):
                gr.Markdown("Upload a CSV file with a `text` column.")
                with gr.Row():
                    multi_csv_in = gr.File(label="Upload CSV", file_types=[".csv"])
                    multi_csv_out = gr.File(label="Download Results")
                multi_csv_btn = gr.Button("Process CSV", variant="secondary")

            with gr.Accordion("Load Custom Model", open=False):
                gr.Markdown(
                    "Load a finetuned multiclass model from a local directory "
                    "to use instead of the default pretrained classifier."
                )
                multi_model_path = gr.Textbox(
                    label="Model directory path",
                    placeholder="e.g., ./finetuned_model",
                )
                with gr.Row():
                    multi_load_btn = gr.Button("Load Model", variant="secondary")
                    multi_reset_btn = gr.Button(
                        "Reset to Pretrained", variant="secondary",
                    )
                multi_status = gr.Markdown("")

        # ================================================================
        # QUESTION ANSWERING TAB
        # ================================================================
        with gr.Tab("Question Answering"):
            gr.Markdown(info_callout(
                "Extract answers from a context passage. Provide a paragraph of "
                "text and ask a question about it. Every sufficiently-confident "
                "answer span is returned, ranked by confidence and labelled."
            ))
            with gr.Row(equal_height=True):
                with gr.Column():
                    qa_context = gr.Textbox(
                        lines=6,
                        placeholder="Paste the context passage here...",
                        label="Context",
                    )
                    qa_question = gr.Textbox(
                        lines=2,
                        placeholder="What would you like to know?",
                        label="Question",
                    )
                    qa_btn = gr.Button("Get Answer", variant="primary")
                with gr.Column():
                    qa_output = gr.HTML(label="Answer")
                    qa_label_output = gr.HTML(label="Infrastructure Label")

            with gr.Accordion("Batch Processing (CSV)", open=False):
                gr.Markdown(
                    "Upload a CSV file with `context` and `question` columns "
                    "to process multiple questions at once."
                )
                with gr.Row():
                    qa_csv_in = gr.File(
                        label="Upload CSV", file_types=[".csv"],
                    )
                    qa_csv_out = gr.File(label="Download Results")
                qa_csv_btn = gr.Button("Process CSV", variant="secondary")

        # ================================================================
        # FINE-TUNE TAB
        # ================================================================
        with gr.Tab("Fine-tune"):
            gr.Markdown(info_callout(
                "Fine-tune a binary or multiclass classifier on your own data. "
                "Upload labeled TSV files, pick a base model, and train. "
                "Or compare multiple models head-to-head on the same dataset."
            ))

            # -- Data --
            gr.Markdown("### Data")
            gr.Markdown(
                "TSV files, no header, format: `text[TAB]label` "
                "(binary: 0/1, multiclass: 0, 1, 2, ...)"
            )
            with gr.Row():
                ft_ex_binary_btn = gr.Button(
                    "Load Example: Binary", variant="secondary", size="sm",
                )
                ft_ex_multi_btn = gr.Button(
                    "Load Example: Multiclass (4 classes)", variant="secondary", size="sm",
                )
            with gr.Row():
                ft_train_file = gr.File(
                    label="Train", file_types=[".tsv", ".csv", ".txt"],
                )
                ft_dev_file = gr.File(
                    label="Validation", file_types=[".tsv", ".csv", ".txt"],
                )
                ft_test_file = gr.File(
                    label="Test", file_types=[".tsv", ".csv", ".txt"],
                )

            # -- Configuration --
            gr.Markdown("### Configuration")
            with gr.Row():
                ft_task = gr.Radio(
                    ["Binary", "Multiclass"],
                    label="Task Type", value="Binary",
                )
                ft_model = gr.Dropdown(
                    choices=list(FINETUNE_MODELS.keys()),
                    label="Base Model",
                    value=list(FINETUNE_MODELS.keys())[0],
                )
            with gr.Row():
                ft_epochs = gr.Number(
                    label="Epochs", value=3, minimum=1, maximum=100, precision=0,
                )
                ft_batch = gr.Number(
                    label="Batch Size", value=8, minimum=1, maximum=128, precision=0,
                )
                ft_lr = gr.Number(
                    label="Learning Rate", value=2e-5, minimum=1e-7, maximum=1e-2,
                )

            with gr.Accordion("Advanced Settings", open=False):
                with gr.Row():
                    ft_weight_decay = gr.Number(
                        label="Weight Decay", value=0.01, minimum=0, maximum=1,
                    )
                    ft_warmup = gr.Number(
                        label="Warmup Ratio", value=0.1, minimum=0, maximum=0.5,
                    )
                    ft_max_len = gr.Number(
                        label="Max Sequence Length", value=512,
                        minimum=32, maximum=8192, precision=0,
                    )
                with gr.Row():
                    ft_grad_accum = gr.Number(
                        label="Gradient Accumulation", value=1,
                        minimum=1, maximum=64, precision=0,
                    )
                    ft_fp16 = gr.Checkbox(
                        label="Mixed Precision (FP16)", value=False,
                    )
                    ft_patience = gr.Number(
                        label="Early Stopping Patience", value=3,
                        minimum=0, maximum=20, precision=0,
                    )
                ft_scheduler = gr.Dropdown(
                    ["linear", "cosine", "constant", "constant_with_warmup"],
                    label="LR Scheduler", value="linear",
                )
                gr.Markdown("**Parameter-Efficient Fine-Tuning (PEFT)**")
                with gr.Row():
                    ft_use_lora = gr.Checkbox(
                        label="Use LoRA", value=False,
                    )
                    ft_lora_rank = gr.Number(
                        label="LoRA Rank (r)", value=8,
                        minimum=1, maximum=256, precision=0,
                    )
                    ft_lora_alpha = gr.Number(
                        label="LoRA Alpha", value=16,
                        minimum=1, maximum=512, precision=0,
                    )
                    ft_use_qlora = gr.Checkbox(
                        label="QLoRA (4-bit, CUDA only)", value=False,
                    )

            # -- Train --
            ft_train_btn = gr.Button(
                "Start Training", variant="primary", size="lg",
            )

            # State for the trained model
            ft_model_state = gr.State(None)
            ft_tokenizer_state = gr.State(None)
            ft_num_labels_state = gr.State(None)

            with gr.Accordion("Training Log", open=False) as ft_log_accordion:
                ft_log = gr.Textbox(
                    lines=12, interactive=False, elem_classes="log-output",
                    show_label=False,
                )

            # -- Results + Try Model (hidden until training completes) --
            with gr.Column(visible=False) as ft_results_col:
                gr.Markdown("### Results")
                with gr.Row(equal_height=True):
                    with gr.Column(scale=2):
                        ft_metrics = gr.Dataframe(
                            label="Test Set Metrics",
                            headers=["Metric", "Score"],
                            interactive=False,
                        )
                    with gr.Column(scale=3):
                        gr.Markdown("**Try your model**")
                        ft_try_input = gr.Textbox(
                            lines=2, label="Input Text",
                            placeholder="Type text to classify...",
                        )
                        with gr.Row():
                            ft_try_btn = gr.Button("Predict", variant="primary")
                        ft_try_output = gr.HTML(label="Prediction")

            # -- Save + Batch (hidden until training completes) --
            with gr.Column(visible=False) as ft_actions_col:
                with gr.Row(equal_height=True):
                    with gr.Column():
                        gr.Markdown("**Download model**")
                        ft_save_btn = gr.Button("Prepare Download", variant="secondary")
                        ft_save_file = gr.File(label="Download Model (.zip)")
                        ft_save_status = gr.Markdown("")
                    with gr.Column():
                        gr.Markdown("**Batch predictions**")
                        ft_batch_in = gr.File(
                            label="Upload CSV (needs 'text' column)",
                            file_types=[".csv"],
                        )
                        ft_batch_btn = gr.Button(
                            "Run Predictions", variant="secondary",
                        )
                        ft_batch_out = gr.File(label="Download Results")

            # -- Compare Models --
            gr.Markdown("---")
            with gr.Accordion("Compare Multiple Models", open=False):
                gr.Markdown(
                    "Train the same dataset on different base models and compare "
                    "performance side by side. Uses the data and task type above."
                )
                cmp_models = gr.CheckboxGroup(
                    choices=list(FINETUNE_MODELS.keys()),
                    label="Select models to compare (pick 2 or more)",
                )
                with gr.Row():
                    cmp_epochs = gr.Number(label="Epochs", value=3, minimum=1, precision=0)
                    cmp_batch = gr.Number(label="Batch Size", value=8, minimum=1, precision=0)
                    cmp_lr = gr.Number(label="Learning Rate", value=2e-5, minimum=1e-7)
                with gr.Row():
                    cmp_use_lora = gr.Checkbox(label="Use LoRA", value=False)
                    cmp_lora_rank = gr.Number(label="LoRA Rank", value=8, minimum=1, maximum=256, precision=0)
                    cmp_lora_alpha = gr.Number(label="LoRA Alpha", value=16, minimum=1, maximum=512, precision=0)
                cmp_btn = gr.Button("Compare Models", variant="primary")
                cmp_log = gr.Textbox(
                    label="Comparison Log", lines=8,
                    interactive=False, elem_classes="log-output",
                )
                with gr.Column(visible=False) as cmp_results_col:
                    cmp_table = gr.Dataframe(
                        label="Comparison Results", interactive=False,
                    )
                    cmp_plot = gr.Plot(label="Metrics Comparison")
                    cmp_roc = gr.Plot(label="ROC Curves")

        # ================================================================
        # ACTIVE LEARNING TAB
        # ================================================================
        with gr.Tab("Active Learning"):
            gr.Markdown(info_callout(
                "**Active learning** iteratively selects the most uncertain "
                "samples from an unlabeled pool for you to label, then retrains. "
                "This lets you build a strong classifier with far fewer labels."
            ))

            # -- Data --
            gr.Markdown("### Data")
            gr.Markdown(
                "**Seed file** — small labeled set (TSV, `text[TAB]label`).  \n"
                "**Pool file** — unlabeled texts (one per line, or CSV with `text` column).  \n"
                "**Dev file** *(optional)* — held-out labeled set to track metrics."
            )
            al_ex_btn = gr.Button(
                "Load Example: Binary Active Learning",
                variant="secondary", size="sm",
            )
            with gr.Row():
                al_seed_file = gr.File(
                    label="Labeled Seed (TSV)",
                    file_types=[".tsv", ".csv", ".txt"],
                )
                al_pool_file = gr.File(
                    label="Unlabeled Pool",
                    file_types=[".tsv", ".csv", ".txt"],
                )
                al_dev_file = gr.File(
                    label="Dev / Validation (optional)",
                    file_types=[".tsv", ".csv", ".txt"],
                )

            # -- Configuration --
            gr.Markdown("### Configuration")
            with gr.Row():
                al_task = gr.Radio(
                    ["Binary", "Multiclass"],
                    label="Task Type", value="Binary",
                )
                al_model_dd = gr.Dropdown(
                    choices=list(FINETUNE_MODELS.keys()),
                    label="Base Model",
                    value=list(FINETUNE_MODELS.keys())[0],
                )
            with gr.Row():
                al_strategy = gr.Dropdown(
                    ["entropy", "margin", "least_confidence"],
                    label="Query Strategy", value="entropy",
                )
                al_query_size = gr.Number(
                    label="Samples per Round", value=20,
                    minimum=1, maximum=500, precision=0,
                )
            with gr.Row():
                al_epochs = gr.Number(
                    label="Epochs per Round", value=3,
                    minimum=1, maximum=50, precision=0,
                )
                al_batch_size = gr.Number(
                    label="Batch Size", value=8,
                    minimum=1, maximum=128, precision=0,
                )
                al_lr = gr.Number(
                    label="Learning Rate", value=2e-5,
                    minimum=1e-7, maximum=1e-2,
                )
            with gr.Accordion("Advanced", open=False):
                with gr.Row():
                    al_max_len = gr.Number(
                        label="Max Sequence Length", value=512,
                        minimum=32, maximum=8192, precision=0,
                    )
                    al_use_lora = gr.Checkbox(label="Use LoRA", value=False)
                    al_lora_rank = gr.Number(
                        label="LoRA Rank", value=8,
                        minimum=1, maximum=256, precision=0,
                    )
                    al_lora_alpha = gr.Number(
                        label="LoRA Alpha", value=16,
                        minimum=1, maximum=512, precision=0,
                    )

            al_init_btn = gr.Button(
                "Initialize Active Learning", variant="primary", size="lg",
            )

            # -- State --
            al_state = gr.State({})
            al_model_state = gr.State(None)
            al_tokenizer_state = gr.State(None)

            with gr.Accordion("Log", open=False):
                al_log = gr.Textbox(
                    lines=12, interactive=False, elem_classes="log-output",
                    show_label=False,
                )

            # -- Annotation panel (hidden until init) --
            with gr.Column(visible=False) as al_annotation_col:
                gr.Markdown("### Label These Samples")
                gr.Markdown(
                    "Fill in the **Label** column with integer class labels "
                    "(e.g. 0 or 1 for binary). Then click **Submit**."
                )
                al_annotation_df = gr.Dataframe(
                    headers=["Text", "Label"],
                    interactive=True,
                )
                with gr.Row():
                    al_submit_btn = gr.Button(
                        "Submit Labels & Next Round",
                        variant="primary",
                    )

                al_chart = gr.Plot(label="Metrics Across Rounds")

                gr.Markdown("### Download Model")
                with gr.Row():
                    al_save_btn = gr.Button("Prepare Download", variant="secondary")
                    al_save_file = gr.File(label="Download Model (.zip)")
                    al_save_status = gr.Markdown("")

    # ---- FOOTER ----
    gr.Markdown(
        "<div style='text-align: center; padding: 1rem 0; margin-top: 0.5rem; "
        "border-top: 1px solid #e5e7eb;'>"
        "<p style='color: #888; font-size: 0.85rem; margin: 0;'>"
        "Developed by "
        "<a href='http://shreyasmeher.com' target='_blank' "
        "style='color: #ff6b35; text-decoration: none;'>Shreyas Meher</a>"
        "</p>"
        "<p style='color: #999; font-size: 0.75rem; margin: 0.5rem 0 0; "
        "max-width: 700px; margin-left: auto; margin-right: auto; line-height: 1.4;'>"
        "If you use ConfliBERT in your research, please cite:<br>"
        "<em>Brandt, P.T., Alsarra, S., D'Orazio, V., Heintze, D., Khan, L., "
        "Meher, S., Osorio, J. and Sianan, M., 2025. Extractive versus Generative "
        "Language Models for Political Conflict Text Classification. "
        "Political Analysis, pp.1&ndash;29.</em>"
        "</p></div>"
    )

    # ====================================================================
    # EVENT HANDLERS
    # ====================================================================

    # NER
    ner_btn.click(
        fn=named_entity_recognition, inputs=[ner_input], outputs=[ner_output],
    )
    ner_csv_btn.click(
        fn=process_csv_ner, inputs=[ner_csv_in], outputs=[ner_csv_out],
    )

    # Binary Classification
    clf_btn.click(
        fn=text_classification,
        inputs=[clf_input, custom_clf_model, custom_clf_tokenizer],
        outputs=[clf_output],
    )
    clf_csv_btn.click(
        fn=process_csv_binary,
        inputs=[clf_csv_in, custom_clf_model, custom_clf_tokenizer],
        outputs=[clf_csv_out],
    )
    clf_load_btn.click(
        fn=load_custom_model,
        inputs=[clf_model_path],
        outputs=[custom_clf_model, custom_clf_tokenizer, clf_status],
    )
    clf_reset_btn.click(
        fn=reset_custom_model,
        outputs=[custom_clf_model, custom_clf_tokenizer, clf_status],
    )

    # Multilabel Classification
    multi_btn.click(
        fn=multilabel_classification,
        inputs=[multi_input, custom_multi_model, custom_multi_tokenizer],
        outputs=[multi_output],
    )
    multi_csv_btn.click(
        fn=process_csv_multilabel, inputs=[multi_csv_in], outputs=[multi_csv_out],
    )
    multi_load_btn.click(
        fn=load_custom_model,
        inputs=[multi_model_path],
        outputs=[custom_multi_model, custom_multi_tokenizer, multi_status],
    )
    multi_reset_btn.click(
        fn=reset_custom_model,
        outputs=[custom_multi_model, custom_multi_tokenizer, multi_status],
    )

    # Question Answering
    qa_btn.click(
        fn=question_answering,
        inputs=[qa_context, qa_question],
        outputs=[qa_output, qa_label_output],
    )
    qa_csv_btn.click(
        fn=process_csv_qa, inputs=[qa_csv_in], outputs=[qa_csv_out],
    )

    # Fine-tuning: example dataset loaders
    ft_ex_binary_btn.click(
        fn=load_example_binary,
        outputs=[ft_train_file, ft_dev_file, ft_test_file, ft_task],
    )
    ft_ex_multi_btn.click(
        fn=load_example_multiclass,
        outputs=[ft_train_file, ft_dev_file, ft_test_file, ft_task],
    )

    # Fine-tuning: training
    ft_train_btn.click(
        fn=run_finetuning,
        inputs=[
            ft_train_file, ft_dev_file, ft_test_file,
            ft_task, ft_model,
            ft_epochs, ft_batch, ft_lr,
            ft_weight_decay, ft_warmup, ft_max_len,
            ft_grad_accum, ft_fp16, ft_patience, ft_scheduler,
            ft_use_lora, ft_lora_rank, ft_lora_alpha, ft_use_qlora,
        ],
        outputs=[
            ft_log, ft_metrics,
            ft_model_state, ft_tokenizer_state, ft_num_labels_state,
            ft_results_col, ft_actions_col,
        ],
        concurrency_limit=1,
    )

    # Try finetuned model
    ft_try_btn.click(
        fn=predict_finetuned,
        inputs=[ft_try_input, ft_model_state, ft_tokenizer_state, ft_num_labels_state],
        outputs=[ft_try_output],
    )

    # Save finetuned model
    ft_save_btn.click(
        fn=save_finetuned_model,
        inputs=[ft_model_state, ft_tokenizer_state],
        outputs=[ft_save_file, ft_save_status],
    )

    # Batch predictions with finetuned model
    ft_batch_btn.click(
        fn=batch_predict_finetuned,
        inputs=[ft_batch_in, ft_model_state, ft_tokenizer_state, ft_num_labels_state],
        outputs=[ft_batch_out],
    )

    # Active Learning: example loader
    al_ex_btn.click(
        fn=load_example_active_learning,
        outputs=[al_seed_file, al_pool_file, al_dev_file, al_task],
    )

    # Active Learning
    al_init_btn.click(
        fn=al_initialize,
        inputs=[
            al_seed_file, al_pool_file, al_dev_file,
            al_task, al_model_dd, al_strategy, al_query_size,
            al_epochs, al_batch_size, al_lr, al_max_len,
            al_use_lora, al_lora_rank, al_lora_alpha,
        ],
        outputs=[
            al_state, al_model_state, al_tokenizer_state,
            al_annotation_df, al_log, al_chart,
            al_annotation_col,
        ],
        concurrency_limit=1,
    )

    al_submit_btn.click(
        fn=al_submit_and_continue,
        inputs=[
            al_annotation_df, al_state, al_model_state, al_tokenizer_state,
            al_log,
        ],
        outputs=[
            al_state, al_model_state, al_tokenizer_state,
            al_annotation_df, al_log, al_chart,
        ],
        concurrency_limit=1,
    )

    al_save_btn.click(
        fn=al_save_model,
        inputs=[al_model_state, al_tokenizer_state],
        outputs=[al_save_file, al_save_status],
    )

    # Model comparison
    cmp_btn.click(
        fn=run_comparison,
        inputs=[
            ft_train_file, ft_dev_file, ft_test_file,
            ft_task, cmp_models, cmp_epochs, cmp_batch, cmp_lr,
            cmp_use_lora, cmp_lora_rank, cmp_lora_alpha,
        ],
        outputs=[cmp_log, cmp_table, cmp_plot, cmp_roc, cmp_results_col],
        concurrency_limit=1,
    )


# ============================================================================
# LAUNCH
# ============================================================================

demo.launch(share=True, ssr_mode=False)
