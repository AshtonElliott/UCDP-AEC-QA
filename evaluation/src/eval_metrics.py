import re
import numpy as np
from scipy.optimize import linear_sum_assignment

class TokenSpanEvaluator:
    def __init__(self, iou_threshold=0.5):
        self.threshold = iou_threshold

    def get_token_indices(self, span, full_text):
        # converts raw character boundaries into a set of distinct token indices
        start, end = span.get('start'), span.get('end')
        if start is None or end is None or not full_text:
            return set()
            
        token_indices = set()
        for token_idx, match in enumerate(re.finditer(r'\b\w+\b', full_text.lower())):
            if match.start() >= start and match.end() <= end:
                token_indices.add(token_idx)
        return token_indices

    def calculate_iou(self, gold_set, pred_set):
        # calculate token-level IOU
        if not gold_set and not pred_set:
            return 1.0  
        if not gold_set or not pred_set:
            return 0.0  
        
        intersection = len(gold_set.intersection(pred_set))
        union = len(gold_set.union(pred_set))
        return intersection / union

    def evaluate_document_spans(self, gold_spans, pred_spans, full_text):
        # computes the optimal bipartite matching matrix and returns TP, FP, FN, IoU, and EM counts
        num_gold = len(gold_spans)
        num_pred = len(pred_spans)
        entry_tp = 0
        entry_em = 0  # track exact token matches

        # if both human and model correctly agreed there are NO answers
        if num_gold == 0 and num_pred == 0:
            return 0, 0, 0, 1.0, 1  # 1 perfect exact match document

        if num_gold > 0 and num_pred > 0:
            iou_matrix = np.zeros((num_gold, num_pred))
            
            for g_idx, g_span in enumerate(gold_spans):
                g_tokens = self.get_token_indices(g_span, full_text)
                
                for p_idx, p_span in enumerate(pred_spans):
                    p_tokens = self.get_token_indices(p_span, full_text)
                    iou_matrix[g_idx, p_idx] = self.calculate_iou(g_tokens, p_tokens)
            
            # hungarian algorithm matching
            gold_ind, pred_ind = linear_sum_assignment(-iou_matrix)
            for g_idx, p_idx in zip(gold_ind, pred_ind):
                if iou_matrix[g_idx, p_idx] >= self.threshold:
                    entry_tp += 1
                if iou_matrix[g_idx, p_idx] == 1.0:
                    entry_em += 1

        entry_fp = num_pred - entry_tp
        entry_fn = num_gold - entry_tp
        
        # calculate overall document-level token overlap
        whole_gold = set().union(*[self.get_token_indices(g, full_text) for g in gold_spans]) if gold_spans else set()
        whole_pred = set().union(*[self.get_token_indices(p, full_text) for p in pred_spans]) if pred_spans else set()
        doc_iou = self.calculate_iou(whole_gold, whole_pred)

        return entry_tp, entry_fp, entry_fn, doc_iou, entry_em