import re
import sys
import string
import numpy as np
from bert_score import score
from collections import Counter
import warnings
import torch

warnings.filterwarnings("ignore")

class EvaluationEngine:    
    @staticmethod
    def normalize_answer(s):
        """Official SQuAD 2.0 string normalization."""
        if not s: return ""
        def remove_articles(text): return re.sub(r'\b(a|an|the)\b', ' ', text)
        def white_space_fix(text): return ' '.join(text.split())
        def remove_punc(text):
            exclude = set(string.punctuation)
            return ''.join(ch for ch in text if ch not in exclude)
        return white_space_fix(remove_articles(remove_punc(s.lower())))

    @classmethod
    def deduplicate_texts(cls, texts):
        seen = set()
        deduped = []
        for t in texts:
            norm = cls.normalize_answer(t)
            if norm and norm not in seen:
                seen.add(norm)
                deduped.append(t)
        return deduped

    @classmethod
    def evaluate_strict_tuple_match(cls, gold_spans, pred_spans):
        """
        Label F1 / Labeled Span F1: evaluates (text, label) pairs as inseparable units for one document.
        Returns document-level Precision, Recall, and F1.
        """
        # Return P, R, F1
        if not gold_spans and not pred_spans: return 1.0, 1.0, 1.0
        if not gold_spans or not pred_spans: return 0.0, 0.0, 0.0
        
        def get_normalized_tuples(spans):
            tuples = []
            for s in spans:
                text = cls.normalize_answer(s.get('text', ''))
                if not text: 
                    continue
                labels = s.get('labels', [])
                lbl = labels[0].strip().lower() if isinstance(labels, list) and labels else "no_label"
                tuples.append((text, lbl))
            return tuples

        g_counts = Counter(get_normalized_tuples(gold_spans))
        p_counts = Counter(get_normalized_tuples(pred_spans))
        
        tp = sum((g_counts & p_counts).values())
        fp = sum((p_counts - g_counts).values())
        fn = sum((g_counts - p_counts).values())
        
        if tp == 0:
            return 0.0, 0.0, 0.0
            
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        f1 = (2 * precision * recall) / (precision + recall)
        
        return precision, recall, f1

    @classmethod
    def compute_squad_exact(cls, a_gold, a_pred):
        """Official SQuAD 2.0 Exact Match for a single gold/pred string pair."""
        return int(cls.normalize_answer(a_gold) == cls.normalize_answer(a_pred))

    @classmethod
    def compute_squad_f1(cls, a_gold, a_pred):
        """Official SQuAD 2.0 Token F1 for a single gold/pred string pair."""
        gold_toks = cls.normalize_answer(a_gold).split()
        pred_toks = cls.normalize_answer(a_pred).split()
        common = Counter(gold_toks) & Counter(pred_toks)
        num_same = sum(common.values())
        
        if len(gold_toks) == 0 or len(pred_toks) == 0:
            return int(gold_toks == pred_toks)
        if num_same == 0:
            return 0.0
            
        precision = 1.0 * num_same / len(pred_toks)
        recall = 1.0 * num_same / len(gold_toks)
        return (2 * precision * recall) / (precision + recall)

    @classmethod
    def evaluate_ie_squad_metrics(cls, g_texts, p_texts):
        """
        Multi-span IE lexical scores after SQuAD normalization.

        Returns:
            prec, rec, set_text_f1: Precision, Recall, and F1 over exact normalized phrase multisets.
            token_f1: max-mean official SQuAD token F1 across span pairs.
        """
        # 1. Fast-fail on empty lists (Return P, R, F1, TokenF1)
        if not g_texts and not p_texts: return 1.0, 1.0, 1.0, 1.0
        if not g_texts or not p_texts: return 0.0, 0.0, 0.0, 0.0

        # 2. Normalize and filter out strings that normalize to ""
        g_norm = [cls.normalize_answer(g) for g in g_texts if cls.normalize_answer(g)]
        p_norm = [cls.normalize_answer(p) for p in p_texts if cls.normalize_answer(p)]

        # 3. Guard against lists that became empty after normalization
        if not g_norm and not p_norm: return 1.0, 1.0, 1.0, 1.0
        if not g_norm or not p_norm: return 0.0, 0.0, 0.0, 0.0
        
        g_counts = Counter(g_norm)
        p_counts = Counter(p_norm)
        
        tp = sum((g_counts & p_counts).values())
        fp = sum((p_counts - g_counts).values())
        fn = sum((g_counts - p_counts).values())
        
        if tp == 0:
            prec, rec, set_text_f1 = 0.0, 0.0, 0.0
        else:
            prec = tp / (tp + fp)
            rec = tp / (tp + fn)
            set_text_f1 = (2 * prec * rec) / (prec + rec)

        # Token F1 (official SQuAD bag-of-tokens F1, max-mean over span pairs)
        p_f1_scores = [max(cls.compute_squad_f1(g, p) for g in g_norm) for p in p_norm]
        r_f1_scores = [max(cls.compute_squad_f1(g, p) for p in p_norm) for g in g_norm]

        p_f1 = float(np.mean(p_f1_scores)) if p_f1_scores else 0.0
        r_f1 = float(np.mean(r_f1_scores)) if r_f1_scores else 0.0
        token_f1 = 2 * (p_f1 * r_f1) / (p_f1 + r_f1) if (p_f1 + r_f1) > 0 else 0.0

        return round(prec, 4), round(rec, 4), round(set_text_f1, 4), round(token_f1, 4)

    @classmethod
    def _bipartite_bertscore_f1(cls, g_texts, p_texts):
        """Max-mean BERTScore F1 over pred×gold pairs. Returns a single F1."""
        if not g_texts and not p_texts:
            return 1.0
        if not g_texts or not p_texts:
            return 0.0

        norm_p = [cls.normalize_answer(p) for p in p_texts]
        norm_g = [cls.normalize_answer(g) for g in g_texts]
        norm_p = [p for p in norm_p if p]
        norm_g = [g for g in norm_g if g]
        if not norm_p and not norm_g:
            return 1.0
        if not norm_p or not norm_g:
            return 0.0

        cands = [p for p in norm_p for _ in norm_g]
        refs = [g for _ in norm_p for g in norm_g]
        flat = cls.run_global_bertscore_backend(cands, refs)
        num_p, num_g = len(norm_p), len(norm_g)
        mat = np.array([flat[r * num_g:(r + 1) * num_g] for r in range(num_p)])
        mp = float(np.clip(mat.max(axis=1).mean(), 0.0, 1.0))
        mr = float(np.clip(mat.max(axis=0).mean(), 0.0, 1.0))
        return float(np.clip(round(2 * (mp * mr) / (mp + mr) if (mp + mr) > 0 else 0.0, 4), 0.0, 1.0))

    @classmethod
    def evaluate_bipartite_bertscore(cls, g_texts, p_texts):
        """Per-doc standard BERTScore. Returns (unused, unused, bertscore_f1) for older callers."""
        bertscore_f1 = cls._bipartite_bertscore_f1(g_texts, p_texts)
        return 0.0, 0.0, bertscore_f1


    @classmethod
    def run_global_bertscore_backend(cls, cands, refs):
        """
        Processes global candidate-reference sentence pairs on GPU using 
        Unique Pair Caching (Memoization) to prevent redundant DeBERTa forward passes.
        """
        num_pairs = len(cands)
        if num_pairs == 0: 
            return []

        # 1. Map all pairs to unique candidate-reference combinations
        unique_pairs = []
        pair_to_unique_idx = {}
        
        for c, r in zip(cands, refs):
            pair_key = (c, r)
            if pair_key not in pair_to_unique_idx:
                pair_to_unique_idx[pair_key] = len(unique_pairs)
                unique_pairs.append(pair_key)

        unique_cands = [p[0] for p in unique_pairs]
        unique_refs = [p[1] for p in unique_pairs]

        # 2. Run BERTScore ONCE on the deduplicated pair list
        try:
            target_device = "cuda" if torch.cuda.is_available() else "cpu"
            current_batch_size = min(512, len(unique_pairs))
            unique_f1_scores = []

            while current_batch_size >= 1:
                try:
                    if target_device == "cuda":
                        torch.cuda.empty_cache()
                    _, _, F1_tensor = score(
                        unique_cands, unique_refs, 
                        model_type="microsoft/deberta-large-mnli", 
                        lang="en", 
                        rescale_with_baseline=True, 
                        device=target_device, 
                        batch_size=current_batch_size, 
                        verbose=False
                    )
                    unique_f1_scores = F1_tensor.tolist()
                    break 
                except RuntimeError as e:
                    if "out of memory" in str(e).lower() and current_batch_size > 1:
                        current_batch_size = max(1, current_batch_size // 2)
                        print(f"\n[HPC Notice] Scaling GPU batch size down to {current_batch_size}...", file=sys.stderr)
                        continue
                    else:
                        raise e

            # 3. Reconstruct full sequence mapped back to original indices
            if unique_f1_scores:
                return [unique_f1_scores[pair_to_unique_idx[(c, r)]] for c, r in zip(cands, refs)]

        except Exception as e:
            print(f"\n[CUDA ERROR] Global BERT-Score execution failed: {str(e)}", file=sys.stderr)

        return [0.0] * num_pairs