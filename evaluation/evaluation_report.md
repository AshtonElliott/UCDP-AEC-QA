

```text

--- Starting Correlation Pipeline ---
loading human rater sheets...

===============================================================================================
Total Framework Samples: 100
Comparison Table (Spearman Rank Matrix):
-----------------------------------------------------------------------------------------------
  Evaluation Dimension | Spearman rho (ρ) | p-value
-----------------------------------------------------------------------------------------------
  SQuAD Exact Match    |           0.8490 | 6.67525e-29
  DeBERTa Precision    |           0.9388 | 3.75725e-47
  DeBERTa Recall       |           0.9226 | 2.49267e-42
  DeBERTa F1-Score     |           0.9549 | 1.85201e-53
-----------------------------------------------------------------------------------------------

LLM Comparison Summary (Sorted by Avg_BS_F1):
-----------------------------------------------------------------------------------------------
                     Rank  Samples_Evaluated  Avg_Norm_Human_Score  Avg_SQuAD_EM  Avg_BS_F1  Avg_BS_Precision  Avg_BS_Recall
True_Model_Identity                                                                                                         
Qwen 3.8B               1                 20                 0.279         0.283      0.358             0.414          0.326
Mistral                 2                 20                 0.300         0.235      0.320             0.320          0.382
Llama 3.1               3                 20                 0.275         0.212      0.277             0.362          0.264
Gemma 4.e4B             4                 20                 0.183         0.187      0.250             0.195          0.402
Gemma 3.4B              5                 20                 0.121         0.095      0.163             0.144          0.213
-----------------------------------------------------------------------------------------------

Side-by-Side Ranking Comparison:
-----------------------------------------------------------------------------------------------
Rank Human Expert Preference Machine Metric Preference (Avg_BS_F1)
  #1         Mistral (0.300)                     Qwen 3.8B (0.358)
  #2       Qwen 3.8B (0.279)                       Mistral (0.320)
  #3       Llama 3.1 (0.275)                     Llama 3.1 (0.277)
  #4     Gemma 4.e4B (0.183)                   Gemma 4.e4B (0.250)
  #5      Gemma 3.4B (0.121)                    Gemma 3.4B (0.163)
===============================================================================================

--- Starting IAA Analysis ---
found 3 grading sheets. Merging...
calculating metrics across 100 graded articles...

=================================================================
Total Overlapping Articles: 100
1. Krippendorff's Alpha (Ordinal):   0.8772
2. ICC(2,k) (Absolute Agreement):    0.9557
-----------------------------------------------------------------
=================================================================

--- Starting Comprehensive Evaluation ---
loading human sheets...
isolated 50 rows.

calculating IAA...
running evaluation metrics...



================================================================================
Row Number: 50

Human Baseline Agreement:
  • Krippendorff's Alpha:           0.7676
  • Intraclass Correlation ICC(2,k): 0.9098

Human-Model Comparison:
--------------------------------------------------------------------------------
  Metric                     | Spearman rho (ρ) | p-value
--------------------------------------------------------------------------------
  SQuAD Exact Match (Strict) |           0.6160 | 1.92076e-06
  DeBERTa F1 (Semantic)      |           0.7369 | 1.04964e-09
  DeBERTa Precision          |           0.6401 | 5.57615e-07
  DeBERTa Recall             |           0.5385 | 5.47925e-05
--------------------------------------------------------------------------------

================================================================================
 Delta Analysis (Top 5 Disagreements):
================================================================================

--- Disagreement #1 (Score Delta: 0.6702) ---
ID           : EVAL_073
Ground Truth : tanks and other heavy weapons
AI Prediction: tanks | heavy weapons
Scores       : Human = 0.9167 | BERTScore = 0.2465

--- Disagreement #2 (Score Delta: 0.3818) ---
ID           : EVAL_009
Ground Truth : warplanes shell | warplanes | airstrikes | airstrikes | warplanes | barrel bombs | warplanes | warplanes | rocket shells | aerial and ground bombardment | shelling | aerial bombardment | barrel bombs | warplanes | ground shelling | shelling | airstrikes | shelling | shelling | aerial bombardment | rocket shelling | shells | shelling | shells
AI Prediction: warplanes | warplanes | warplanes | warplanes | warplanes | warplanes | warplanes | airstrikes | airstrikes | airstrikes | barrel bombs | barrel bombs | rocket shells | ground shelling | targeting | targeting | targeting | targeting
Scores       : Human = 0.2500 | BERTScore = 0.6318

--- Disagreement #3 (Score Delta: 0.3334) ---
ID           : EVAL_071
Ground Truth : airstrike | airstrike | airstrike
AI Prediction: airstrike | airstrike | airstrike | militants | militants | armed
Scores       : Human = 0.3333 | BERTScore = 0.6667

--- Disagreement #4 (Score Delta: 0.3248) ---
ID           : EVAL_012
Ground Truth : sniper fire | shelling and aerial bombardment | barrel bombs | sniper fire | shelling and aerial attacks | missile | sniping and shelling | sniper | shelling | barrel bomb | barrel bombs | barrel bombs | bomb | Shelling | barrel bombs | barrel bombs
AI Prediction: shelling | shelling | shelling | shelling | Shelling | sniping | barrel bombs | barrel bombs | barrel bombs | barrel bombs | barrel bombs | bombardment | bombardment | dropping barrel bombs | dropping barrel bombs | fighting | fighting
Scores       : Human = 0.2500 | BERTScore = 0.5748

--- Disagreement #5 (Score Delta: 0.3144) ---
ID           : EVAL_065
Ground Truth : Airstrikes | IED | landmine | warplanes bombed | artillery shells | shelling
AI Prediction: Airstrikes | explosion | explosion | explosion | shelling | shelling | IED | mortar shells | landmine | warplanes | artillery shells
Scores       : Human = 0.3333 | BERTScore = 0.6477

================================================================================


```
