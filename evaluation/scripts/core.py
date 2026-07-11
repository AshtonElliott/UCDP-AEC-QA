import re
import string
import numpy as np
from scipy.optimize import linear_sum_assignment
from bert_score import score
from collections import Counter
import warnings

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

    # ==========================================
    # STRICT ENTITY METRICS (Text + Label)
    # ==========================================
    @classmethod
    def evaluate_strict_entity_match(cls, gold_spans, pred_spans):
        """Standard Information Extraction Tuple Matcher: (text, label)"""
        def get_tuples(spans):
            tuples = []
            for s in spans:
                text = s.get('text', '').strip().lower() 
                if not text: continue
                labels = s.get('labels', [])
                lbl = labels[0] if isinstance(labels, list) and len(labels) > 0 else "NO_LABEL"
                tuples.append((text, lbl))
            return tuples

        g_tuples = get_tuples(gold_spans)
        p_tuples = get_tuples(pred_spans)
        
        g_counts = Counter(g_tuples)
        p_counts = Counter(p_tuples)
        
        tp = sum((g_counts & p_counts).values())
        fp = sum((p_counts - g_counts).values())
        fn = sum((g_counts - p_counts).values())
        
        return tp, fp, fn

    # ==========================================
    # STRICT TEXT METRICS (Span Only)
    # ==========================================
    @classmethod
    def evaluate_exact_match(cls, g_texts, p_texts):
        if not g_texts and not p_texts: return 1.0
        if not g_texts or not p_texts: return 0.0

        exact_matches = 0
        normalized_preds = [cls.normalize_answer(p) for p in p_texts]
        
        matched_preds = set()
        for g in g_texts:
            norm_gold = cls.normalize_answer(g)
            for i, p in enumerate(normalized_preds):
                if i not in matched_preds and norm_gold == p:
                    exact_matches += 1
                    matched_preds.add(i)
                    break
                    
        return exact_matches / max(len(g_texts), len(p_texts))

    @classmethod
    def evaluate_bipartite_bertscore(cls, g_texts, p_texts):
        if not g_texts and not p_texts: return 1.0, 1.0, 1.0
        if not g_texts or not p_texts: return 0.0, 0.0, 0.0

        norm_p_texts = [cls.normalize_answer(p) for p in p_texts]
        norm_g_texts = [cls.normalize_answer(g) for g in g_texts]
        
        cands = [p for p in norm_p_texts for g in norm_g_texts]
        refs = [g for p in norm_p_texts for g in norm_g_texts]
                
        try:
            _, _, F1_tensor = score(cands, refs, model_type="microsoft/deberta-large-mnli", lang="en", rescale_with_baseline=True, device="cpu", batch_size=4, verbose=False)
            flat_f1_scores = F1_tensor.tolist()
            num_p, num_g = len(norm_p_texts), len(norm_g_texts)
            matrix = [flat_f1_scores[i * num_g:(i + 1) * num_g] for i in range(num_p)]
            
            row_ind, col_ind = linear_sum_assignment(-np.array(matrix))
            matched_score_sum = sum([matrix[r][c] for r, c in zip(row_ind, col_ind)])
            
            mean_p = matched_score_sum / num_p
            mean_r = matched_score_sum / num_g
            final_f1 = 2 * (mean_p * mean_r) / (mean_p + mean_r) if (mean_p + mean_r) > 0 else 0.0
            
            return float(np.clip(round(mean_p, 4), 0.0, 1.0)), float(np.clip(round(mean_r, 4), 0.0, 1.0)), float(np.clip(round(final_f1, 4), 0.0, 1.0))
        except Exception:
            return 0.0, 0.0, 0.0

    # ==========================================
    # RELAXED METRICS (Deduplication + Partial)
    # ==========================================
    @classmethod
    def evaluate_dedup_bertscore(cls, g_texts, p_texts):
        g_dedup = cls.deduplicate_texts(g_texts)
        p_dedup = cls.deduplicate_texts(p_texts)
        return cls.evaluate_bipartite_bertscore(g_dedup, p_dedup)

    @classmethod
    def evaluate_iou_match(cls, g_texts, p_texts):
        g_dedup = cls.deduplicate_texts(g_texts)
        p_dedup = cls.deduplicate_texts(p_texts)
        
        if not g_dedup and not p_dedup: return 1.0
        if not g_dedup or not p_dedup: return 0.0

        norm_p = [cls.normalize_answer(p) for p in p_dedup]
        norm_g = [cls.normalize_answer(g) for g in g_dedup]

        matrix = np.zeros((len(norm_p), len(norm_g)))
        
        for i, p in enumerate(norm_p):
            p_tokens = set(p.split())
            for j, g in enumerate(norm_g):
                g_tokens = set(g.split())
                
                if not p_tokens and not g_tokens:
                    matrix[i, j] = 1.0
                elif not p_tokens or not g_tokens:
                    matrix[i, j] = 0.0
                else:
                    intersection = len(p_tokens.intersection(g_tokens))
                    union = len(p_tokens.union(g_tokens))
                    matrix[i, j] = intersection / union

        row_ind, col_ind = linear_sum_assignment(-matrix)
        matched_score_sum = sum([matrix[r][c] for r, c in zip(row_ind, col_ind)])
        
        mean_p = matched_score_sum / len(norm_p)
        mean_r = matched_score_sum / len(norm_g)
        final_iou_f1 = 2 * (mean_p * mean_r) / (mean_p + mean_r) if (mean_p + mean_r) > 0 else 0.0
        
        return float(np.clip(round(final_iou_f1, 4), 0.0, 1.0))