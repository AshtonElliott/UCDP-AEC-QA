# ============================================================================
# ConfliBERT QA Fine-tuning (Q1: Weapons & Methods, Q2: Damaged Infrastructure)
#
# Fine-tunes one of the PyTorch SQuAD 2.0 checkpoints from appQA.py / appQAQ2.py
# on your own labelled data, using the same windowing (QA_WINDOW / stride) and
# the same [CLS]-as-null convention that the inference code uses for abstention.
#
# Input: canonical JSON, the same format process_json_qa reads:
#   [{"id": ..., "source_article" | "context": "...",
#     "question1" | "question2" | "question": "...",
#     "answer_labels": [{"text": "...", "start": 12, "end": 30, "labels": [...]}],
#     "no_answer": "..."}   # genuine no-answer records have no answer_labels
#   ]
#
# Only the QA (span extraction) model is fine-tuned. The Q2 infrastructure
# labels ('Energy', 'Water', ...) are assigned afterwards by appQAQ2's lexicon +
# zero-shot classifier; the "labels" in your ground truth are ignored here.
#
# Usage:
#   python finetuneQA.py --question 1 --train-data q1_labels.json --base-model 2
#   python finetuneQA.py --question 2 --train-data q2_labels.json --base-model 3
#
# Afterwards, call register_finetuned_model(<output dir>, question=1|2) before
# running inference, then re-tune tau/alpha with tune_cv.py: the defaults in the
# app modules were calibrated for the original checkpoints, not yours.
# ============================================================================

import os
import sys
import json
import inspect
import argparse
import importlib

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForQuestionAnswering,
    TrainingArguments,
    Trainer,
    default_data_collator,
)


# ============================================================================
# PER-QUESTION CONFIGURATION
# ============================================================================

# Each question maps to the app module that runs inference for it, the JSON
# keys that module reads the question from (first present wins, mirroring its
# process_json_qa), and the names of its per-model tables.
QUESTION_CONFIG = {
    1: {
        'module': 'appQA',
        'question_keys': ('question', 'question1'),
        'slugs': 'MODEL_FILE_SLUGS_Q1',
        'defaults': 'MODEL_TUNED_DEFAULTS_Q1',
        'slug_suffix': '_results.json',
    },
    2: {
        'module': 'appQAQ2',
        'question_keys': ('question2', 'question'),
        'slugs': 'MODEL_FILE_SLUGS_Q2',
        'defaults': 'MODEL_TUNED_DEFAULTS_Q2',
        'slug_suffix': '_results2.json',
    },
}


def get_app(question):
    """Import and return the inference module for question 1 or 2."""
    if int(question) not in QUESTION_CONFIG:
        raise ValueError(f"question must be one of {sorted(QUESTION_CONFIG)}, got {question!r}")
    return importlib.import_module(QUESTION_CONFIG[int(question)]['module'])


# ============================================================================
# DATA LOADING
# ============================================================================

def load_training_docs(path, question_keys):
    """Canonical JSON -> list of documents with their gold answer spans.

    Each document is {'id', 'context', 'question', 'spans': [(start, end), ...]}.
    A record with no valid spans is a negative (no-answer) document.
    """
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    docs, skipped_spans = [], 0
    for n, item in enumerate(data):
        context = item.get('source_article', item.get('context', ''))
        question = ''
        for key in question_keys:
            if item.get(key):
                question = item[key]
                break
        if not context or not question:
            continue
        context, question = str(context), str(question)

        spans = set()
        for s in item.get('answer_labels') or []:
            start, end = s.get('start'), s.get('end')
            if (not isinstance(start, int) or not isinstance(end, int)
                    or not 0 <= start < end <= len(context)):
                skipped_spans += 1
                continue
            spans.add((start, end))
        docs.append({'id': str(item.get('id', n)), 'context': context,
                     'question': question, 'spans': sorted(spans)})

    if skipped_spans:
        print(f"[Warning] Skipped {skipped_spans} answer span(s) with invalid offsets.")
    return docs


def split_by_document(docs, val_frac, seed):
    """Split documents into (train, val, val_ids) by id so windows never leak."""
    ids = sorted({d['id'] for d in docs})
    rng = np.random.default_rng(seed)
    rng.shuffle(ids)
    n_val = max(1, int(round(len(ids) * val_frac))) if val_frac > 0 else 0
    val_ids = set(ids[:n_val])
    train = [d for d in docs if d['id'] not in val_ids]
    val = [d for d in docs if d['id'] in val_ids]
    return train, val, sorted(val_ids)


# ============================================================================
# FEATURE CREATION
# ============================================================================

class QAFeatures(Dataset):
    def __init__(self, features):
        self.features = features

    def __len__(self):
        return len(self.features)

    def __getitem__(self, i):
        return {k: torch.tensor(v) for k, v in self.features[i].items()}


def _window_stride(app, tokenizer, question):
    """Stride used at inference for this question, or None if it cannot be windowed.

    appQAQ2 shrinks the stride when a long question leaves little room for
    context, and refuses questions that leave less than QA_MIN_CONTEXT. appQA
    uses a fixed QA_STRIDE. Training mirrors whichever applies.
    """
    min_context = getattr(app, 'QA_MIN_CONTEXT', None)
    if min_context is None:
        return app.QA_STRIDE
    n_special = tokenizer.num_special_tokens_to_add(pair=True)
    n_question = len(tokenizer(question, add_special_tokens=False)['input_ids'])
    room = app.QA_WINDOW - n_question - n_special
    if room < min_context:
        return None
    return min(app.QA_STRIDE, room // 2)


def build_features(docs, tokenizer, app):
    """Tokenize documents into windowed training features with start/end labels.

    For each window, every gold span that fits fully inside it becomes one
    feature (label = that span's first/last token). A window containing no gold
    span gets a single [CLS] feature (index 0), the same position the inference
    code reads as the null score. A window is never labelled null when it holds
    a gold span, which matters for multi-answer documents: otherwise the span
    that lives in another window would teach the model to abstain on windows
    that contain a different, perfectly valid answer.
    """
    if not tokenizer.is_fast:
        raise ValueError("A fast tokenizer is required (offset mappings are used).")

    features, skipped_docs = [], 0
    keep_keys = ('input_ids', 'attention_mask', 'token_type_ids')
    for doc in docs:
        stride = _window_stride(app, tokenizer, doc['question'])
        if stride is None:
            skipped_docs += 1
            continue
        enc = tokenizer(
            doc['question'], doc['context'],
            truncation='only_second',
            max_length=app.QA_WINDOW,
            stride=stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding='max_length',
        )
        for i, offsets in enumerate(enc['offset_mapping']):
            seq_ids = enc.sequence_ids(i)
            ctx_tokens = [t for t, s in enumerate(seq_ids) if s == 1]
            if not ctx_tokens:
                continue
            first, last = ctx_tokens[0], ctx_tokens[-1]
            base = {k: enc[k][i] for k in keep_keys if k in enc}

            labels = []
            for s_char, e_char in doc['spans']:
                if offsets[first][0] <= s_char and offsets[last][1] >= e_char:
                    ts, te = first, last
                    while ts <= te and offsets[ts][0] <= s_char:
                        ts += 1
                    while te >= ts and offsets[te][1] >= e_char:
                        te -= 1
                    labels.append((ts - 1, te + 1))
            for start_pos, end_pos in (labels or [(0, 0)]):
                features.append({**base, 'start_positions': start_pos,
                                 'end_positions': end_pos})

    if skipped_docs:
        print(f"[Warning] Skipped {skipped_docs} document(s) whose question leaves "
              f"under {app.QA_MIN_CONTEXT} tokens of context per window.")
    return QAFeatures(features)


# ============================================================================
# TRAINING
# ============================================================================

def _resolve_base_model(app, base_model):
    """Accept '1'-'7', a QA_MODELS display name, a HF id, or a local path."""
    name = app.MODEL_KEY_TO_NAME.get(str(base_model), base_model)
    model_id = app.QA_MODELS.get(name, name)
    if model_id == app.Salsarra_QA_model_name:
        raise ValueError(
            "Model 1 (salsarra/ConfliBERT-QA) only ships TensorFlow weights and "
            "cannot be fine-tuned with this PyTorch script. Choose a SQuAD 2.0 "
            "model (2-7) or any PyTorch QA checkpoint / local path."
        )
    return name, model_id


def finetune_qa(train_json, base_model, output_dir, question=1, epochs=3, lr=3e-5,
                batch_size=8, grad_accum=2, val_frac=0.15, seed=42,
                warmup_ratio=0.1, weight_decay=0.01):
    """Fine-tune a QA model for question 1 or 2 and save it to output_dir."""
    app = get_app(question)
    cfg = QUESTION_CONFIG[int(question)]
    base_name, base_id = _resolve_base_model(app, base_model)
    print(f"Question {question} | base model: {base_name} ({base_id})")

    docs = load_training_docs(train_json, cfg['question_keys'])
    if not docs:
        raise ValueError(
            f"No usable records in {train_json} (needs a context and one of "
            f"{cfg['question_keys']})."
        )
    train_docs, val_docs, val_ids = split_by_document(docs, val_frac, seed)
    n_neg = sum(not d['spans'] for d in train_docs)
    n_spans = sum(len(d['spans']) for d in train_docs)
    print(f"Documents: {len(train_docs)} train ({n_neg} no-answer, {n_spans} gold spans), "
          f"{len(val_docs)} validation")

    tokenizer = AutoTokenizer.from_pretrained(base_id)
    model = AutoModelForQuestionAnswering.from_pretrained(base_id)

    train_ds = build_features(train_docs, tokenizer, app)
    val_ds = build_features(val_docs, tokenizer, app) if val_docs else None
    print(f"Features (windows): {len(train_ds)} train, {len(val_ds) if val_ds else 0} validation")

    # transformers 4.40-4.x argument names changed across minor versions.
    ta_params = inspect.signature(TrainingArguments.__init__).parameters
    eval_key = 'eval_strategy' if 'eval_strategy' in ta_params else 'evaluation_strategy'
    args_kwargs = dict(
        output_dir=output_dir,
        num_train_epochs=epochs,
        learning_rate=lr,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size * 2,
        gradient_accumulation_steps=grad_accum,
        weight_decay=weight_decay,
        warmup_ratio=warmup_ratio,
        fp16=torch.cuda.is_available(),
        save_strategy='epoch',
        save_total_limit=1,
        logging_steps=25,
        seed=seed,
        report_to='none',
    )
    if val_ds is not None:
        args_kwargs[eval_key] = 'epoch'
        args_kwargs['load_best_model_at_end'] = True
        args_kwargs['metric_for_best_model'] = 'eval_loss'
        args_kwargs['greater_is_better'] = False
    training_args = TrainingArguments(**args_kwargs)

    trainer_kwargs = dict(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=default_data_collator,
    )
    if 'processing_class' in inspect.signature(Trainer.__init__).parameters:
        trainer_kwargs['processing_class'] = tokenizer
    else:
        trainer_kwargs['tokenizer'] = tokenizer
    trainer = Trainer(**trainer_kwargs)
    trainer.train()

    os.makedirs(output_dir, exist_ok=True)
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    info = {
        'question': int(question),
        'base_model': base_id,
        'train_data': os.path.abspath(train_json),
        'epochs': epochs, 'learning_rate': lr, 'batch_size': batch_size,
        'grad_accum': grad_accum, 'val_frac': val_frac, 'seed': seed,
        'window': app.QA_WINDOW, 'stride': app.QA_STRIDE,
        'held_out_document_ids': val_ids,
    }
    with open(os.path.join(output_dir, 'finetune_info.json'), 'w', encoding='utf-8') as f:
        json.dump(info, f, indent=2)
    print(f"Fine-tuned model saved to: {output_dir}")
    print("Held-out document ids are in finetune_info.json; evaluate on those "
          "(or a separate test set), not on the training data.")
    return output_dir


# ============================================================================
# USING THE FINE-TUNED MODEL WITH appQA.py / appQAQ2.py
# ============================================================================

def register_finetuned_model(model_dir, question=1, display_name=None,
                             tau=None, alpha=None):
    """Make a saved model selectable in appQA (question=1) or appQAQ2 (question=2).

    Call before get_qa_model / process_json_qa of that module. tau and alpha are
    placeholders (the module's untuned defaults) until you tune them with
    tune_cv.py for this model. Returns the display name to pass as
    model_display_name.
    """
    app = get_app(question)
    cfg = QUESTION_CONFIG[int(question)]
    display_name = display_name or f"ConfliBERT Q{int(question)} fine-tuned"
    slug = "".join(c if c.isalnum() or c in "-_" else "_" for c in display_name)
    app.QA_MODELS[display_name] = os.path.abspath(model_dir)
    getattr(app, cfg['slugs'])[display_name] = f"{slug}{cfg['slug_suffix']}"
    getattr(app, cfg['defaults'])[display_name] = {
        "tau": app.QA_NULL_THRESHOLD if tau is None else tau,
        "alpha": app.QA_ANSWER_MIN_RATIO if alpha is None else alpha,
    }
    return display_name


# ============================================================================
# COMMAND LINE
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune ConfliBERT QA on labelled JSON data")
    parser.add_argument("--question", type=int, choices=sorted(QUESTION_CONFIG), default=1,
                        help="1 = Weapons & Methods (appQA), 2 = Damaged Infrastructure (appQAQ2)")
    parser.add_argument("--train-data", required=True,
                        help="Canonical JSON with answer_labels (start/end char offsets)")
    parser.add_argument("--base-model", default="2",
                        help="1-7, a QA_MODELS name, a HF model id, or a local path (default: 2). "
                             "Model 1 is TensorFlow-only and not supported.")
    parser.add_argument("--output-dir", default=None,
                        help="Where to save the model (default: finetuned_q<question>)")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--grad-accum", type=int, default=2,
                        help="Gradient accumulation steps (effective batch = batch-size x this)")
    parser.add_argument("--val-frac", type=float, default=0.15,
                        help="Fraction of documents held out for validation (0 disables)")
    parser.add_argument("--seed", type=int, default=42)
    cli = parser.parse_args()

    finetune_qa(
        train_json=cli.train_data,
        base_model=cli.base_model,
        output_dir=cli.output_dir or os.path.join(SCRIPT_DIR, f"finetuned_q{cli.question}"),
        question=cli.question,
        epochs=cli.epochs,
        lr=cli.lr,
        batch_size=cli.batch_size,
        grad_accum=cli.grad_accum,
        val_frac=cli.val_frac,
        seed=cli.seed,
    )
