import re
import string
import numpy as np
from scipy.optimize import linear_sum_assignment
from bert_score import score
import warnings

warnings.filterwarnings("ignore")

class EvaluationEngine:    
    @staticmethod
    def normalize_answer(s):
        if not s:
            return ""
        def remove_articles(text):
            return re.sub(r'\b(a|an|the)\b', ' ', text)
        def white_space_fix(text):
            return ' '.join(text.split())
        def remove_punc(text):
            exclude = set(string.punctuation)
            return ''.join(ch for ch in text if ch not in exclude)
        return white_space_fix(remove_articles(remove_punc(s.lower())))

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

        # text normalization
        norm_p_texts = [cls.normalize_answer(p) for p in p_texts]
        norm_g_texts = [cls.normalize_answer(g) for g in g_texts]
        
        cands = [p for p in norm_p_texts for g in norm_g_texts]
        refs = [g for p in norm_p_texts for g in norm_g_texts]
                
        try:
            # BERTScore calculation
            _, _, F1_tensor = score(cands, refs, model_type="microsoft/deberta-large-mnli", lang="en", rescale_with_baseline=True, device="cpu", batch_size=4, verbose=False)
            flat_f1_scores = F1_tensor.tolist()
            num_p, num_g = len(norm_p_texts), len(norm_g_texts)
            matrix = [flat_f1_scores[i * num_g:(i + 1) * num_g] for i in range(num_p)]
            
            # 1-to-1 bipartite locking
            row_ind, col_ind = linear_sum_assignment(-np.array(matrix))
            matched_score_sum = sum([matrix[r][c] for r, c in zip(row_ind, col_ind)])
            
            mean_p = matched_score_sum / num_p
            mean_r = matched_score_sum / num_g
            final_f1 = 2 * (mean_p * mean_r) / (mean_p + mean_r) if (mean_p + mean_r) > 0 else 0.0
            
            return float(np.clip(round(mean_p, 4), 0.0, 1.0)), float(np.clip(round(mean_r, 4), 0.0, 1.0)), float(np.clip(round(final_f1, 4), 0.0, 1.0))
        except Exception as e:
            return 0.0, 0.0, 0.0