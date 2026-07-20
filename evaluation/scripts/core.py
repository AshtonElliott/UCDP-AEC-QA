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
    def evaluate_strict_entity_match(cls, gold_spans, pred_spans):
        """standard information extraction tuple: (text, label)"""
        def get_tuples(spans):
            tuples = []
            for s in spans:
                text = s.get('text', '').strip().lower() 
                if not text: continue
                labels = s.get('labels', [])
                lbl = labels[0] if isinstance(labels, list) and len(labels) > 0 else "NO_LABEL"
                tuples.append((text, lbl))
            return tuples

        g_counts = Counter(get_tuples(gold_spans))
        p_counts = Counter(get_tuples(pred_spans))
        
        tp = sum((g_counts & p_counts).values())
        fp = sum((p_counts - g_counts).values())
        fn = sum((g_counts - p_counts).values())
        return tp, fp, fn

    @classmethod
    def evaluate_exact_match(cls, g_texts, p_texts):
        """O(n) EM using counter intersection"""
        if not g_texts and not p_texts: return 1.0
        if not g_texts or not p_texts: return 0.0

        g_counts = Counter([cls.normalize_answer(g) for g in g_texts])
        p_counts = Counter([cls.normalize_answer(p) for p in p_texts])
        
        exact_matches = sum((g_counts & p_counts).values())
        return exact_matches / max(len(g_texts), len(p_texts))

    @classmethod
    def run_global_bertscore_backend(cls, cands, refs):
        """processes the entire dataset's sentences in highly parallelized global chunks"""
        num_pairs = len(cands)
        if num_pairs == 0: return []
        
        try:
            target_device = "cuda" if torch.cuda.is_available() else "cpu"
            current_batch_size = min(512, num_pairs)
            F1_tensor = None
            
            while current_batch_size >= 1:
                try:
                    if target_device == "cuda":
                        torch.cuda.empty_cache()
                    _, _, F1_tensor = score(
                        cands, refs, 
                        model_type="microsoft/deberta-large-mnli", 
                        lang="en", 
                        rescale_with_baseline=True, 
                        device=target_device, 
                        batch_size=current_batch_size, 
                        verbose=False
                    )
                    break 
                except RuntimeError as e:
                    if "out of memory" in str(e).lower() and current_batch_size > 1:
                        current_batch_size = max(1, current_batch_size // 2)
                        print(f"\n[HPC Notice] Scaling global batch size down to {current_batch_size}...", file=sys.stderr)
                        continue
                    else:
                        raise e

            if F1_tensor is not None:
                return F1_tensor.tolist()
        except Exception as e:
            print(f"\n[CUDA ERROR] Global BERT-Score backend execution failed: {str(e)}", file=sys.stderr)
        return [0.0] * num_pairs

    @classmethod
    def evaluate_iou_match(cls, g_texts, p_texts):
        """optimized greedy token IoU match avoiding NumPy init overhead for strings"""
        g_dedup = cls.deduplicate_texts(g_texts)
        p_dedup = cls.deduplicate_texts(p_texts)
        
        if not g_dedup and not p_dedup: return 1.0
        if not g_dedup or not p_dedup: return 0.0

        norm_p = [set(cls.normalize_answer(p).split()) for p in p_dedup]
        norm_g = [set(cls.normalize_answer(g).split()) for g in g_dedup]

        matrix = []
        for p_tokens in norm_p:
            row = []
            for g_tokens in norm_g:
                if not p_tokens and not g_tokens:
                    row.append(1.0)
                elif not p_tokens or not g_tokens:
                    row.append(0.0)
                else:
                    row.append(len(p_tokens & g_tokens) / len(p_tokens | g_tokens))
            matrix.append(row)

        matrix_np = np.array(matrix)
        mean_p = matrix_np.max(axis=1).mean() if len(norm_p) > 0 else 0.0
        mean_r = matrix_np.max(axis=0).mean() if len(norm_g) > 0 else 0.0
        final_iou_f1 = 2 * (mean_p * mean_r) / (mean_p + mean_r) if (mean_p + mean_r) > 0 else 0.0
        return float(np.clip(round(final_iou_f1, 4), 0.0, 1.0))
