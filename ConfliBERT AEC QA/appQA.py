# ============================================================================
# ConfliBERT - Conflict & Political Violence NLP Toolkit
# University of Texas at Dallas | Event Data Lab
# ============================================================================

import os
import time
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
    AutoModelForQuestionAnswering,
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
    import tensorflow as tf
    import tf_keras   # noqa: F401
    import keras      # noqa: F401
    from transformers import TFAutoModelForQuestionAnswering
    _USE_TF_QA = True
except (ImportError, ModuleNotFoundError):
    print("Using PyTorch for QA model")

import gradio as gr
import numpy as np
import pandas as pd
import re
import csv
import tempfile
import threading
from collections import OrderedDict
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

# SQuAD 2.0 abstention. The model signals "no answer" by scoring the [CLS] token
# highly. A question is abstained on when the [CLS] logit sum exceeds the best
# span's logit sum by more than this margin. Raise it to abstain less often,
# lower it (negative values allowed) to abstain more often.
QA_NULL_THRESHOLD = 0.0


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

# SQuAD 2.0 fine-tunes selectable in the QA tab. All are BERT-base, so the
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

# QA models are loaded on first selection in the QA tab and kept in an LRU cache.
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


get_qa_model(QA_DEFAULT_MODEL)

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
        
def question_answering(context, question, model_display_name=QA_DEFAULT_MODEL):
    if not context or not question:
        return "Please provide both context and question."
    try:
        qa_model, qa_tokenizer = get_qa_model(model_display_name)
        answers = qa_extract_answers(context, question, qa_model, qa_tokenizer)
        if not answers:
            return (
                "<span style='color: #ef4444; font-weight: 600;'>"
                "No answer found in the provided context.</span>"
            )
        rows = [
            "<div style='margin-bottom: 0.4rem;'>"
            f"<span style='color: #6b7280; font-size: 0.8rem;'>{n}.</span> "
            f"<span style='color: #10b981; font-weight: 600;'>{a['answer']}</span> "
            f"<span style='color: #6b7280; font-size: 0.8rem;'>"
            f"({a['score']:.1%})</span></div>"
            for n, a in enumerate(answers, 1)
        ]
        return "<div>" + "".join(rows) + "</div>"
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


# ============================================================================
# CSV BATCH PROCESSING
# ============================================================================

def process_csv_qa(file, model_display_name=QA_DEFAULT_MODEL, progress=gr.Progress()):
    start = time.perf_counter()
    path = get_path(file)
    if path is None:
        return None
    df = pd.read_csv(path)
    if 'context' not in df.columns or 'question' not in df.columns:
        raise ValueError("CSV must contain 'context' and 'question' columns")
    qa_model, qa_tokenizer = get_qa_model(model_display_name)

    answers, confidences, counts = [], [], []
    for _, row in progress.tqdm(df.iterrows(), total=len(df), desc="Answering"):
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
            answers.append(re.sub(r'<[^>]+>', '', handle_error(e)).strip())
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

    out = tempfile.NamedTemporaryFile(suffix='_qa_results.csv', delete=False)
    df.to_csv(out.name, index=False)

    # Print finshing time
    runtime = time.perf_counter() - start
    print(f"QA finished: {runtime:.1f}s ({runtime / 60:.1f} min)",flush=True)
    return out.name

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
        # QUESTION ANSWERING TAB
        # ================================================================
        with gr.Tab("Question Answering"):
            gr.Markdown(info_callout(
                "Extract answers from a context passage. Provide a paragraph of "
                "text and ask a question about it. Every sufficiently-confident "
                "answer span is returned, ranked by confidence."
            ))
            qa_model_choice = gr.Dropdown(
                choices=list(QA_MODELS.keys()),
                value=QA_DEFAULT_MODEL,
                label="QA Model",
                info="Applies to single questions and CSV batches. "
                     "A model not used yet is downloaded on first use.",
            )
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
    # Question Answering
    qa_btn.click(
        fn=question_answering,
        inputs=[qa_context, qa_question, qa_model_choice],
        outputs=[qa_output],
    )
    qa_csv_btn.click(
        fn=process_csv_qa,
        inputs=[qa_csv_in, qa_model_choice],
        outputs=[qa_csv_out],
    )
# ============================================================================
# LAUNCH
# ============================================================================

demo.launch(share=True, ssr_mode=False)
