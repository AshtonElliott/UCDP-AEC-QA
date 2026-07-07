

# LLM Evaluation Report

## 1. Inter-Annotator Agreement (Data Quality)


*Based on 96 overlapping articles.*

| Metric | Score |
|---|---|
| **Krippendorff's Alpha** (Ordinal) | 0.9051 |
| **ICC(2,k)** (Absolute Agreement) | 0.9665 |


## 2. Metric-to-Human Correlation


*Total Framework Samples: 96*

### Comparison Table (Spearman Rank Matrix)

| Evaluation Dimension | Spearman rho (ρ) | p-value |
|---|---|---|
| SQuAD Exact Match | 0.7950 | 4.01430e-22 |
| DeBERTa Precision | 0.8136 | 7.29094e-24 |
| DeBERTa Recall | 0.7702 | 4.66481e-20 |
| DeBERTa F1-Score | 0.8178 | 2.80220e-24 |


![Correlation Rank Distribution Plot](assets/pilot_rank_distribution.png)

### LLM Comparison Summary (Sorted by Avg_BS_F1)

| True_Model_Identity   |   Rank |   Samples_Evaluated |   Avg_Norm_Human_Score |   Avg_SQuAD_EM |   Avg_BS_F1 |   Avg_BS_Precision |   Avg_BS_Recall |
|:----------------------|-------:|--------------------:|-----------------------:|---------------:|------------:|-------------------:|----------------:|
| Qwen 3.8B             |      1 |                  19 |                  0.346 |          0.298 |       0.369 |              0.428 |           0.336 |
| Mistral               |      2 |                  19 |                  0.193 |          0.248 |       0.319 |              0.324 |           0.376 |
| Llama 3.1             |      3 |                  19 |                  0.285 |          0.224 |       0.279 |              0.371 |           0.258 |
| Gemma 4.e4B           |      4 |                  20 |                  0.183 |          0.187 |       0.25  |              0.195 |           0.402 |
| Gemma 3.4B            |      5 |                  19 |                  0.118 |          0.1   |       0.167 |              0.149 |           0.207 |

![Pilot Leaderboard](assets/pilot_leaderboard.png)


### Side-by-Side Ranking Comparison

| Rank   | Human Expert Preference   | Machine Metric Preference (Avg_BS_F1)   |
|:-------|:--------------------------|:----------------------------------------|
| #1     | Qwen 3.8B (0.346)         | Qwen 3.8B (0.369)                       |
| #2     | Llama 3.1 (0.285)         | Mistral (0.319)                         |
| #3     | Mistral (0.193)           | Llama 3.1 (0.279)                       |
| #4     | Gemma 4.e4B (0.183)       | Gemma 4.e4B (0.250)                     |
| #5     | Gemma 3.4B (0.118)        | Gemma 3.4B (0.167)                      |

![Pilot Performance Quadrant](assets/pilot_quadrant.png)


## 4. Error Analysis & Top Disagreements


*Evaluated on 45 complex edge-case rows.*

### Human Baseline Agreement (Complex Subset)

| Metric | Score |
|---|---|
| **Krippendorff's Alpha** | 0.8323 |
| **Intraclass Correlation ICC(2,k)** | 0.9380 |

### Human-Model Comparison (Complex Subset)

| Metric | Spearman rho (ρ) | p-value |
|---|---|---|
| **SQuAD Exact Match** (Strict) | 0.7138 | 3.71488e-08 |
| **DeBERTa F1** (Semantic) | 0.8066 | 2.24617e-11 |
| **DeBERTa Precision** | 0.8218 | 4.54450e-12 |
| **DeBERTa Recall** | 0.4777 | 9.05213e-04 |


![Complex Cases Rank Distribution Plot](assets/complex_rank_distribution.png)

### Delta Analysis (Top 5 Disagreements)

**Disagreement #1 (Score Delta: 0.7276)**
* **ID:** EVAL_088
* **Ground Truth:** `airstrike`
* **AI Prediction:** `air strike`
* **Scores:** Human = 1.0000 | BERTScore = 0.2724

---

**Disagreement #2 (Score Delta: 0.4068)**
* **ID:** EVAL_050
* **Ground Truth:** `bombings | shells | shelling | shelling | shelling | shelling | shelling`
* **AI Prediction:** `shells | heavy machineguns | heavy machineguns | shelling | shelling | shelling | shelling | shelling | shelling | bombings`
* **Scores:** Human = 0.4167 | BERTScore = 0.8235

---

**Disagreement #3 (Score Delta: 0.3333)**
* **ID:** EVAL_076
* **Ground Truth:** `ied | ied`
* **AI Prediction:** `IED explosion | joint patrol | joint patrol`
* **Scores:** Human = 0.3333 | BERTScore = 0.0000

---

**Disagreement #4 (Score Delta: 0.2564)**
* **ID:** EVAL_027
* **Ground Truth:** `ied | ied | missiles | shells | shells | shells`
* **AI Prediction:** `missiles | shells | shells | shells | IED | IED | IED`
* **Scores:** Human = 0.6667 | BERTScore = 0.9231

---

**Disagreement #5 (Score Delta: 0.2500)**
* **ID:** EVAL_001
* **Ground Truth:** `bullet`
* **AI Prediction:** `tear gas | bullet | rocks`
* **Scores:** Human = 0.2500 | BERTScore = 0.5000

---

