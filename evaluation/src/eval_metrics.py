import re
import string

class SquadEvaluator:
    # standardizes Exact Match (EM) extraction evaluation using the official SQuAD normalization methodology
    @staticmethod
    def normalize_answer(s):
        # lowercases text, removes punctuation, articles, and extra whitespace
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


    # calculates how many ground truth spans were perfectly extracted.
    def evaluate_exact_match(self, gold_spans, pred_spans):
        
        # both human and model correctly agreed there are no answers (TN)
        if not gold_spans and not pred_spans:
            return 1, 1  
            
        # model hallucinated answers where there were none (FP)
        if not gold_spans and pred_spans:
            return 0, 1 
            
        # model missed all answers (FN)
        if gold_spans and not pred_spans:
            return 0, len(gold_spans)

        exact_matches = 0
        normalized_preds = [self.normalize_answer(p.get('text', '')) for p in pred_spans]
        
        for g_span in gold_spans:
            norm_gold = self.normalize_answer(g_span.get('text', ''))
            # SQuAD checks if the exact normalized string exists in the predictions
            if norm_gold in normalized_preds:
                exact_matches += 1
                
        return exact_matches, len(gold_spans)