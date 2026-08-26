# LLM Evaluation Report
**Total Models Evaluated:** 9
**Total Documents Processed:** 1860
**Full Evaluation Compute Time:** 235.66 seconds

## Part 1: Global Benchmark Leaderboard (SQuAD 2.0 Evaluation)
> *All text metrics formatted as (Overall / HasAns)*

| Model            | Method    |   Doc Count |   Avg Spans |   Abstention (NoAns) |   Missed Answer Rate | Span F1 (Overall / HasAns)   | Labeled Span F1 (Overall / HasAns)   | SQuAD Token F1 (Overall / HasAns)   | BERTScore (Overall / HasAns)   | Deduped BERTScore (Overall / HasAns)   |
|:-----------------|:----------|------------:|------------:|---------------------:|---------------------:|:-----------------------------|:-------------------------------------|:------------------------------------|:-------------------------------|:---------------------------------------|
| Gpt Oss.20B      | Zero-Shot |        1860 |        3.34 |                 0.62 |                 0.04 | 0.53 / 0.44                  | 0.53 / 0.44                          | 0.60 / 0.58                         | 0.61 / 0.60                    | 0.61 / 0.60                            |
|                  | Cookbook  |        1860 |        0.42 |                 0.97 |                 0.81 | 0.49 / 0.01                  | 0.49 / 0.01                          | 0.49 / 0.01                         | 0.50 / 0.01                    | 0.50 / 0.01                            |
| Gemma4.E4B       | Zero-Shot |        1860 |        3.37 |                 0.6  |                 0.03 | 0.52 / 0.44                  | 0.51 / 0.43                          | 0.59 / 0.57                         | 0.59 / 0.58                    | 0.59 / 0.58                            |
|                  | Cookbook  |        1860 |        3.58 |                 0.59 |                 0.05 | 0.51 / 0.42                  | 0.50 / 0.40                          | 0.57 / 0.55                         | 0.58 / 0.56                    | 0.58 / 0.56                            |
| Qwen3.8B         | Zero-Shot |        1860 |        2.98 |                 0.56 |                 0.05 | 0.49 / 0.42                  | 0.49 / 0.42                          | 0.54 / 0.53                         | 0.55 / 0.55                    | 0.55 / 0.55                            |
| Llama3.1.8B      | Zero-Shot |        1860 |        3.53 |                 0.37 |                 0.04 | 0.39 / 0.40                  | 0.38 / 0.38                          | 0.44 / 0.51                         | 0.45 / 0.53                    | 0.45 / 0.53                            |
|                  | Cookbook  |        1860 |        4.62 |                 0.05 |                 0    | 0.22 / 0.39                  | 0.21 / 0.38                          | 0.30 / 0.55                         | 0.30 / 0.56                    | 0.30 / 0.55                            |
| Mistral Nemo.12B | Zero-Shot |        1860 |        2.13 |                 0.57 |                 0.1  | 0.48 / 0.39                  | 0.48 / 0.39                          | 0.53 / 0.50                         | 0.55 / 0.52                    | 0.55 / 0.52                            |
|                  | Cookbook  |        1860 |        2.24 |                 0.58 |                 0.09 | 0.48 / 0.39                  | 0.48 / 0.38                          | 0.54 / 0.49                         | 0.55 / 0.52                    | 0.55 / 0.52                            |
| Gemma3.4B        | Zero-Shot |        1860 |        5.76 |                 0.17 |                 0.02 | 0.25 / 0.34                  | 0.24 / 0.32                          | 0.31 / 0.46                         | 0.32 / 0.48                    | 0.33 / 0.49                            |
|                  | Cookbook  |        1860 |        3.59 |                 0.8  |                 0.18 | 0.56 / 0.32                  | 0.56 / 0.32                          | 0.61 / 0.42                         | 0.62 / 0.44                    | 0.62 / 0.44                            |
| Mistral          | Zero-Shot |        1860 |        2.5  |                 0.73 |                 0.17 | 0.53 / 0.32                  | 0.53 / 0.32                          | 0.57 / 0.41                         | 0.58 / 0.43                    | 0.58 / 0.43                            |
|                  | Cookbook  |        1860 |        2.16 |                 0.69 |                 0.16 | 0.51 / 0.33                  | 0.51 / 0.32                          | 0.56 / 0.43                         | 0.57 / 0.44                    | 0.56 / 0.44                            |
| Vicuna.13B       | Zero-Shot |        1860 |        3.45 |                 0.65 |                 0.25 | 0.43 / 0.20                  | 0.42 / 0.19                          | 0.47 / 0.28                         | 0.46 / 0.27                    | 0.46 / 0.27                            |
|                  | Cookbook  |        1860 |        2.5  |                 0.91 |                 0.67 | 0.47 / 0.02                  | 0.47 / 0.02                          | 0.48 / 0.04                         | 0.47 / 0.04                    | 0.47 / 0.03                            |
| Llama2.13B       | Zero-Shot |        1860 |        1.45 |                 0.46 |                 0.45 | 0.30 / 0.14                  | 0.30 / 0.13                          | 0.32 / 0.18                         | 0.33 / 0.20                    | 0.33 / 0.20                            |

## Part 2: Visual Insights (Zero-Shot Baseline)

### 1. Abstention vs. Extraction Quality
> *Evaluates whether models are 'Ideal Performers' (safe and accurate) or 'Hallucinators' (talkative but unsafe).*
![Master Performance Quadrant](assets/master_quadrant.png)

### 2. Strict vs. Relaxed Evaluation Shift
> *Visualizing the performance penalty models take when evaluated strictly (Span F1) vs. relaxed (Token/Semantic).*
![Strict vs Relaxed](assets/strict_vs_relaxed.png)

### 3. Verbosity vs. Semantic Accuracy
> *Tracking whether models artificially inflate their extraction scores by over-generating spans.*
![Verbosity vs Semantic Accuracy](assets/verbosity_vs_accuracy.png)


---

## Part 3: Task Complexity Breakdown

### Performance Degradation Heatmaps (Zero-Shot Baseline)
> *Overall Score (includes easy abstentions) vs. HasAns Score (true extraction capability).*
![Task Complexity (Overall)](assets/task_complexity_heatmap_overall.png)
![Task Complexity (HasAns)](assets/task_complexity_heatmap_hasans.png)


### Question 1 Leaderboard
| Model            | Method    |   Doc Count |   Avg Spans |   Abstention (NoAns) |   Missed Answer Rate | Span F1 (Overall / HasAns)   | Labeled Span F1 (Overall / HasAns)   | SQuAD Token F1 (Overall / HasAns)   | BERTScore (Overall / HasAns)   | Deduped BERTScore (Overall / HasAns)   |
|:-----------------|:----------|------------:|------------:|---------------------:|---------------------:|:-----------------------------|:-------------------------------------|:------------------------------------|:-------------------------------|:---------------------------------------|
| Gpt Oss.20B      | Zero-Shot |         954 |        5.87 |                 0.02 |                 0.01 | 0.37 / 0.45                  | 0.37 / 0.45                          | 0.48 / 0.59                         | 0.50 / 0.62                    | 0.50 / 0.61                            |
|                  | Cookbook  |         954 |        0.81 |                 0.88 |                 0.78 | 0.18 / 0.01                  | 0.18 / 0.01                          | 0.18 / 0.01                         | 0.19 / 0.02                    | 0.19 / 0.02                            |
| Gemma4.E4B       | Zero-Shot |         954 |        6.01 |                 0.02 |                 0    | 0.36 / 0.44                  | 0.36 / 0.44                          | 0.47 / 0.58                         | 0.48 / 0.60                    | 0.48 / 0.59                            |
|                  | Cookbook  |         954 |        6.45 |                 0    |                 0    | 0.35 / 0.43                  | 0.35 / 0.43                          | 0.46 / 0.57                         | 0.47 / 0.58                    | 0.47 / 0.59                            |
| Mistral Nemo.12B | Zero-Shot |         954 |        3.53 |                 0.04 |                 0.04 | 0.35 / 0.43                  | 0.35 / 0.43                          | 0.43 / 0.53                         | 0.46 / 0.56                    | 0.46 / 0.56                            |
|                  | Cookbook  |         954 |        3.73 |                 0.03 |                 0.03 | 0.35 / 0.43                  | 0.35 / 0.43                          | 0.43 / 0.53                         | 0.46 / 0.56                    | 0.46 / 0.57                            |
| Qwen3.8B         | Zero-Shot |         954 |        4.88 |                 0.03 |                 0.03 | 0.35 / 0.43                  | 0.35 / 0.43                          | 0.43 / 0.53                         | 0.46 / 0.56                    | 0.45 / 0.56                            |
| Llama3.1.8B      | Zero-Shot |         954 |        5.52 |                 0.02 |                 0.02 | 0.33 / 0.41                  | 0.33 / 0.41                          | 0.41 / 0.51                         | 0.43 / 0.54                    | 0.43 / 0.54                            |
|                  | Cookbook  |         954 |        5.62 |                 0    |                 0    | 0.32 / 0.40                  | 0.32 / 0.40                          | 0.44 / 0.55                         | 0.45 / 0.57                    | 0.45 / 0.56                            |
| Mistral          | Zero-Shot |         954 |        4.65 |                 0.06 |                 0.03 | 0.31 / 0.38                  | 0.31 / 0.38                          | 0.40 / 0.48                         | 0.41 / 0.50                    | 0.42 / 0.51                            |
|                  | Cookbook  |         954 |        3.79 |                 0.13 |                 0.05 | 0.32 / 0.37                  | 0.32 / 0.37                          | 0.41 / 0.48                         | 0.43 / 0.50                    | 0.42 / 0.50                            |
| Gemma3.4B        | Zero-Shot |         954 |        9.05 |                 0.01 |                 0.01 | 0.27 / 0.33                  | 0.27 / 0.33                          | 0.36 / 0.45                         | 0.38 / 0.47                    | 0.39 / 0.49                            |
|                  | Cookbook  |         954 |        7    |                 0.01 |                 0    | 0.31 / 0.39                  | 0.31 / 0.39                          | 0.41 / 0.51                         | 0.43 / 0.53                    | 0.43 / 0.53                            |
| Vicuna.13B       | Zero-Shot |         954 |        6.11 |                 0.29 |                 0.21 | 0.22 / 0.20                  | 0.22 / 0.20                          | 0.28 / 0.28                         | 0.28 / 0.27                    | 0.28 / 0.27                            |
|                  | Cookbook  |         954 |        4.87 |                 0.54 |                 0.59 | 0.13 / 0.03                  | 0.13 / 0.03                          | 0.15 / 0.05                         | 0.14 / 0.04                    | 0.14 / 0.04                            |
| Llama2.13B       | Zero-Shot |         954 |        1.67 |                 0.46 |                 0.49 | 0.18 / 0.11                  | 0.18 / 0.11                          | 0.20 / 0.14                         | 0.22 / 0.15                    | 0.22 / 0.16                            |

#### Plot A: Text Extraction Precision & Recall (Q1 - Zero-Shot)
> *This isolates reading comprehension: Did the model locate the correct phrases? HasAns-only bars at exact span match — one operating point per model.*
![Plot A: Text Spans](assets/q1_pr_bars.png)


### Question 2 Leaderboard
| Model            | Method    |   Doc Count |   Avg Spans |   Abstention (NoAns) |   Missed Answer Rate | Span F1 (Overall / HasAns)   | Labeled Span F1 (Overall / HasAns)   | SQuAD Token F1 (Overall / HasAns)   | BERTScore (Overall / HasAns)   | Deduped BERTScore (Overall / HasAns)   |
|:-----------------|:----------|------------:|------------:|---------------------:|---------------------:|:-----------------------------|:-------------------------------------|:------------------------------------|:-------------------------------|:---------------------------------------|
| Gpt Oss.20B      | Zero-Shot |         906 |        0.67 |                 0.77 |                 0.21 | 0.70 / 0.39                  | 0.70 / 0.36                          | 0.73 / 0.53                         | 0.73 / 0.52                    | 0.73 / 0.52                            |
|                  | Cookbook  |         906 |        0.02 |                 1    |                 0.97 | 0.82 / 0.00                  | 0.82 / 0.00                          | 0.82 / 0.00                         | 0.82 / 0.00                    | 0.82 / 0.00                            |
| Gemma4.E4B       | Zero-Shot |         906 |        0.58 |                 0.75 |                 0.17 | 0.69 / 0.40                  | 0.68 / 0.34                          | 0.71 / 0.52                         | 0.71 / 0.51                    | 0.71 / 0.51                            |
|                  | Cookbook  |         906 |        0.55 |                 0.75 |                 0.24 | 0.67 / 0.35                  | 0.66 / 0.27                          | 0.70 / 0.46                         | 0.69 / 0.46                    | 0.69 / 0.46                            |
| Gemma3.4B        | Zero-Shot |         906 |        2.29 |                 0.21 |                 0.08 | 0.24 / 0.38                  | 0.22 / 0.25                          | 0.26 / 0.48                         | 0.26 / 0.49                    | 0.26 / 0.50                            |
|                  | Cookbook  |         906 |        0    |                 1    |                 1    | 0.82 / 0.00                  | 0.82 / 0.00                          | 0.82 / 0.00                         | 0.82 / 0.00                    | 0.82 / 0.00                            |
| Llama3.1.8B      | Zero-Shot |         906 |        1.43 |                 0.46 |                 0.13 | 0.45 / 0.39                  | 0.42 / 0.26                          | 0.47 / 0.49                         | 0.47 / 0.50                    | 0.47 / 0.49                            |
|                  | Cookbook  |         906 |        3.57 |                 0.06 |                 0.02 | 0.12 / 0.37                  | 0.10 / 0.28                          | 0.14 / 0.52                         | 0.14 / 0.51                    | 0.14 / 0.50                            |
| Qwen3.8B         | Zero-Shot |         906 |        0.98 |                 0.69 |                 0.13 | 0.63 / 0.36                  | 0.62 / 0.33                          | 0.66 / 0.51                         | 0.65 / 0.49                    | 0.65 / 0.49                            |
| Llama2.13B       | Zero-Shot |         906 |        1.22 |                 0.47 |                 0.26 | 0.44 / 0.30                  | 0.42 / 0.21                          | 0.45 / 0.39                         | 0.45 / 0.41                    | 0.45 / 0.41                            |
| Mistral Nemo.12B | Zero-Shot |         906 |        0.64 |                 0.71 |                 0.35 | 0.62 / 0.22                  | 0.61 / 0.18                          | 0.64 / 0.33                         | 0.64 / 0.32                    | 0.64 / 0.32                            |
|                  | Cookbook  |         906 |        0.68 |                 0.72 |                 0.38 | 0.63 / 0.20                  | 0.62 / 0.17                          | 0.65 / 0.30                         | 0.65 / 0.29                    | 0.65 / 0.30                            |
| Vicuna.13B       | Zero-Shot |         906 |        0.65 |                 0.75 |                 0.44 | 0.65 / 0.20                  | 0.63 / 0.13                          | 0.66 / 0.28                         | 0.66 / 0.27                    | 0.66 / 0.27                            |
|                  | Cookbook  |         906 |        0    |                 1    |                 1    | 0.82 / 0.00                  | 0.82 / 0.00                          | 0.82 / 0.00                         | 0.82 / 0.00                    | 0.82 / 0.00                            |
| Mistral          | Zero-Shot |         906 |        0.23 |                 0.91 |                 0.82 | 0.76 / 0.07                  | 0.75 / 0.05                          | 0.76 / 0.09                         | 0.76 / 0.09                    | 0.76 / 0.09                            |
|                  | Cookbook  |         906 |        0.44 |                 0.83 |                 0.68 | 0.70 / 0.12                  | 0.70 / 0.08                          | 0.71 / 0.16                         | 0.71 / 0.16                    | 0.71 / 0.16                            |

#### Plot A: Text Extraction Precision & Recall (Q2 - Zero-Shot)
> *This isolates reading comprehension: Did the model locate the correct phrases? HasAns-only bars at exact span match — one operating point per model.*
![Plot A: Text Spans](assets/q2_pr_bars.png)

#### Plot B: Category Mislabeling Breakdown (Q2 - Zero-Shot)
> *Macro-averages hide class-level failures. This 8-panel grid isolates which specific event labels models confuse after extracting the text.*
![Plot B: Category Grid](assets/q2_category_pr_grid.png)


### Question 2: The Classification Penalty (Zero-Shot)
> **Classification Dropoff = set_text_f1 - label_f1**
> *A large gap indicates the model successfully acts as a search engine (finding the correct evidence text) but fails as a classifier (assigning the wrong event label).*
![Classification Dropoff](assets/q2_classification_dropoff.png)


---


## Part 4: Error Analysis & SQuAD 2.0 Edge Cases
> *Automated extraction of specific failure modes across the IE pipeline.*

### Analysis: Question 1

#### A. Missed Extractions (Failed to extract an existing answer)
* **Total cases:** 1761
* **By model:** Vicuna.13B: 617, Gpt Oss.20B: 598, Llama2.13B: 375, Mistral: 63, Mistral Nemo.12B: 57, Qwen3.8B: 26, Llama3.1.8B: 13, Gemma3.4B: 6, Gemma4.E4B: 6

**Edge Case #1 (Gpt Oss.20B)**
* **Ground Truth:** `mine | shelled | shelling | shelling | shelling | shelling | shelling | shelling | shelling | aerial and ground shelling | aerial and ground shelling | ground shelling | ground shelling | bomb | aerial and ground bombardment | airstrikes | airstrikes | airstrikes | airstrikes | airstrikes | airstrikes | airstrikes | shells | shells | barrel bombs | barrel bombs | barrel bombs | bombardment by warplanes | aerial bombardment | aerial bombardment | aerial bombardment | rocket shelling | warplanes | warplanes | warplanes | warplanes | warplanes | warplanes | warplanes | warplanes`
* **Model Prediction:** `[ABSTAINED - Predicted Nothing]`
---

**Edge Case #2 (Llama2.13B)**
* **Ground Truth:** `mine | shelled | shelling | shelling | shelling | shelling | shelling | shelling | shelling | aerial and ground shelling | aerial and ground shelling | ground shelling | ground shelling | bomb | aerial and ground bombardment | airstrikes | airstrikes | airstrikes | airstrikes | airstrikes | airstrikes | airstrikes | shells | shells | barrel bombs | barrel bombs | barrel bombs | bombardment by warplanes | aerial bombardment | aerial bombardment | aerial bombardment | rocket shelling | warplanes | warplanes | warplanes | warplanes | warplanes | warplanes | warplanes | warplanes`
* **Model Prediction:** `[ABSTAINED - Predicted Nothing]`
---

**Edge Case #3 (Vicuna.13B)**
* **Ground Truth:** `shelled | shelled | shelled | shelled | shelled | shelled | shelled | shelled | shelled | barrel bombs | mortars | shelling | shelling | shells | shells | shells | shells | shells | snipers | snipers | snipers | guided missile | drone | mortar shells | bombed | sniper shot | sniper shot | sniper shot | rocket shelling | warplanes | warplanes | warplanes | airstrikes | aerial bombardment | aerial bombardment | aerial bombardment | aerial bombardment | rocket shells`
* **Model Prediction:** `[ABSTAINED - Predicted Nothing]`
---


#### B. Hallucinations (Generated text on an unanswerable article)
* **Total cases:** 2574
* **By model:** Gemma3.4B: 379, Gemma4.E4B: 379, Llama3.1.8B: 378, Mistral Nemo.12B: 369, Mistral: 347, Vicuna.13B: 224, Gpt Oss.20B: 210, Qwen3.8B: 185, Llama2.13B: 103

**Edge Case #1 (Gemma3.4B)**
* **Ground Truth:** `[EMPTY ARTICLE - Should have abstained]`
* **Model Prediction:** `Officer | Officer | officer | Officer | officer | Sergeant | sergeant | Sergeant | Terrorists | terrorists | terrorists | terrorists | terrorists | armed | armed | Armed | armed | armed | Armed | armed | armed | Armed | armed | armed | armed | armed | armed | armed | Armed | armed | armed | group | group | group | group | group | Group | group | group | Group | group | group | group | group | group | group | group | group | group | machineguns | RPGs | RPGs | RPGs | RPGs | launchers | night vision binoculars | sniper rifles | explosive | explosive | explosive | explosive | remote control | ambulance | ambulance`
* **Spans Generated:** 64
---

**Edge Case #2 (Vicuna.13B)**
* **Ground Truth:** `[EMPTY ARTICLE - Should have abstained]`
* **Model Prediction:** `protests | protests | clashes | clashes | clashes | violence | violence | violence | security crackdown | arrests | leaders | leaders | Islamists | Islamists | Islamists | Islamists | Islamists | Morsi | Morsi | Morsi | Morsi | Morsi | Morsi | Morsi | Morsi | Muslim Brotherhood | Muslim Brotherhood | democracy | posters | coup | Coup | Coup | coup | military | military | peaceful rallies | proposals | tensions | prisoners | probe | violence | violence | violence | police | police | police | Police | police | Police | government buildings | stations | Soldiers | thoroughfares | squares | Beltagi | Beltagi | Beltagi | Azhari | parliament | opponents | warrant | inciting violence`
* **Spans Generated:** 62
---

**Edge Case #3 (Llama2.13B)**
* **Ground Truth:** `[EMPTY ARTICLE - Should have abstained]`
* **Model Prediction:** `operation | operation | operation | operation | operation | armed oppositions | terrorists | terrorists | Killed | killed | killed | killed | Afghan National Army | Afghan National Police | operation | operation | operation | operation | operation | armed oppositions | terrorists | terrorists | Killed | killed | killed | killed | wounded | wounded`
* **Spans Generated:** 28
---


#### C. Over-Extraction (High Verbosity, Low Precision)
* **Total cases:** 1520
* **By model:** Vicuna.13B: 350, Gemma3.4B: 347, Llama3.1.8B: 200, Gemma4.E4B: 162, Mistral: 148, Gpt Oss.20B: 113, Mistral Nemo.12B: 90, Qwen3.8B: 70, Llama2.13B: 40

**Edge Case #1 (Vicuna.13B)**
* **Ground Truth:** `Improvised Explosive Devices (IED) borne-vehicle`
* **Model Prediction:** `troops | troops | troops | troops | troops | troops | killed | killed | Boko Haram terrorists | Boko Haram terrorists | Goniri community | Gubio Local Government Area | Yobe | Yobe | Assistant Director | Army Public Relations | 27 Task Force Brigade | insurgents | insurgents | insurgents | insurgents | insurgents | Sambisa Forest | community | community | community | community | engaged | killed | killed | wounded | wounded | high calibre ammunition | fleeing terrorists | Operation DEEDP PUNCH II | attempted to infiltrate | formidable blocking positions | Goniri | Goniri | Gujba Local Government Area | Yobe State | destroyed | neutralised | wounded | wounded | several others | several others | recovered | recovered | long belt of ammunitions | items | unscrupulous elements | society | provided information | insurgents | insurgents | insurgents | insurgents | insurgents | Area | Area | area | utilised the information | made concerted attempt | infiltrate | infiltrate | infiltrate | gallant troops | repelled | attack | attack | attack | crushed them | soldiers | injuries | attack | attack | attack | treatment`
* **Scores:** Spans Generated = 79 | Deduped BERTScore = 0.0000
---

**Edge Case #2 (Gemma3.4B)**
* **Ground Truth:** `airstrikes`
* **Model Prediction:** `UN | un | UN | UN | un | un | un | UN | UN | un | peace plan | peace plan | peace plan | heavy weapons | airstrikes | militant activity | forces | Forces | forces | forces | forces | forces | forces | Forces | forces | forces | forces | coalition | coalition | coalition | coalition | engineering college | port city | military | operation | operation | operation | fighters | soldiers`
* **Scores:** Spans Generated = 39 | Deduped BERTScore = 0.0000
---

**Edge Case #3 (Gpt Oss.20B)**
* **Ground Truth:** `shot | grenades`
* **Model Prediction:** `New Standard Hotel | and | and | and | and | and | and | and | and | and | and | and | and | and | and | and | and | and | and | and | and | and | and | and | and | and | and | and | and`
* **Scores:** Spans Generated = 29 | Deduped BERTScore = 0.0000
---


#### D. Valid Paraphrasing (High Semantic Match, Zero Exact Match)
* **Total cases:** 9
* **By model:** Mistral Nemo.12B: 4, Vicuna.13B: 2, Gemma4.E4B: 1, Mistral: 1, Qwen3.8B: 1

**Edge Case #1 (Gemma4.E4B)**
* **Ground Truth:** `car bomb`
* **Model Prediction:** `car bomb attack`
* **Scores:** Span F1 = 0.0000 | Deduped BERTScore = 0.8099
---

**Edge Case #2 (Vicuna.13B)**
* **Ground Truth:** `bullets`
* **Model Prediction:** `bullet`
* **Scores:** Span F1 = 0.0000 | Deduped BERTScore = 0.7694
---

**Edge Case #3 (Mistral Nemo.12B)**
* **Ground Truth:** `gun`
* **Model Prediction:** `guns`
* **Scores:** Span F1 = 0.0000 | Deduped BERTScore = 0.7508
---


### Analysis: Question 2

#### A. Missed Extractions (Failed to extract an existing answer)
* **Total cases:** 1132
* **By model:** Mistral: 247, Vicuna.13B: 236, Gpt Oss.20B: 194, Gemma3.4B: 177, Mistral Nemo.12B: 120, Gemma4.E4B: 68, Llama2.13B: 43, Llama3.1.8B: 25, Qwen3.8B: 22

**Edge Case #1 (Gemma3.4B)**
* **Ground Truth:** `house | farmlands | Churches | crops | houses | houses | houses | houses | churches`
* **Model Prediction:** `[ABSTAINED - Predicted Nothing]`
---

**Edge Case #2 (Gpt Oss.20B)**
* **Ground Truth:** `house | farmlands | Churches | crops | houses | houses | houses | houses | churches`
* **Model Prediction:** `[ABSTAINED - Predicted Nothing]`
---

**Edge Case #3 (Llama2.13B)**
* **Ground Truth:** `house | farmlands | Churches | crops | houses | houses | houses | houses | churches`
* **Model Prediction:** `[ABSTAINED - Predicted Nothing]`
---


#### B. Hallucinations (Generated text on an unanswerable article)
* **Total cases:** 3655
* **By model:** Llama3.1.8B: 1098, Gemma3.4B: 587, Mistral Nemo.12B: 420, Llama2.13B: 397, Gemma4.E4B: 370, Qwen3.8B: 230, Mistral: 194, Vicuna.13B: 189, Gpt Oss.20B: 170

**Edge Case #1 (Llama3.1.8B)**
* **Ground Truth:** `[EMPTY ARTICLE - Should have abstained]`
* **Model Prediction:** `areas | areas | areas | areas | areas | areas | areas | areas | villages | countryside | countryside | countryside | countryside | countryside | countryside | countryside | countryside | countryside | countryside | countryside | countryside | countryside | countryside | countryside | countryside | countryside | Al-Hbit | Al-Hbit | checkpoint | Abdeen | Abdeen | Horsh Abdeen | Horsh al-Qassabiyyeh | Tremla | Karsaa | Maarrat Al-Sain | Hish | Marayan | Ihsim | Maarrat al-Nu’man | Rowayha | Al-Dana | Jaradah | Mantaf | Ariha | Maarbalit | Shalkh | Ram Hamdan | Kafar Yahmoule | Maarrat Misrin | Haranbush | village | village | Sheikh Bahr | Al-Fiqia | Wadi Al-Deif Camp | Hazanu | Batabu | Killi | killi | Sahl al-Ghab`
* **Spans Generated:** 61
---

**Edge Case #2 (Qwen3.8B)**
* **Ground Truth:** `[EMPTY ARTICLE - Should have abstained]`
* **Model Prediction:** `tunnels | tunnels | tunnels | tunnels | tunnels | trenches | trenches | trenches | trenches | trenches | Baghuz farms | Baghuz farms | Baghuz farms | Baghuz farms | Baghuz farms | Baghuz farms | Baghuz farms | Baghuz farms | Baghuz farms | Baghuz farms | Baghuz farms`
* **Spans Generated:** 21
---

**Edge Case #3 (Gemma3.4B)**
* **Ground Truth:** `[EMPTY ARTICLE - Should have abstained]`
* **Model Prediction:** `road of Damascus | road of Damascus | Brigade 137 | Brigade 137 | Brigade 137 | Brigade 137 | Brigade 137 | gap | gap | gap | gap | gap | gap | gap | gap | booby-trapped cars`
* **Spans Generated:** 16
---


#### C. Over-Extraction (High Verbosity, Low Precision)
* **Total cases:** 91
* **By model:** Llama3.1.8B: 31, Gemma3.4B: 17, Mistral: 9, Qwen3.8B: 9, Mistral Nemo.12B: 7, Llama2.13B: 6, Gpt Oss.20B: 5, Vicuna.13B: 4, Gemma4.E4B: 3

**Edge Case #1 (Mistral)**
* **Ground Truth:** `regime forces checkpoints`
* **Model Prediction:** `other | other | other | other | other | other | other | other | other | other | other | other | other | other | other | other | other | other | other | other | other | other | other | other | other | other | other | other | other | other | other | other | other | other | other | other | other | other | other | other | other | other | other | other | other | other | other | other`
* **Scores:** Spans Generated = 48 | Deduped BERTScore = 0.0000
---

**Edge Case #2 (Llama3.1.8B)**
* **Ground Truth:** `house | house`
* **Model Prediction:** `University housing units | University housing units | University housing units | University housing units | car in Al-Fahama area | car in Al-Fahama area | Madaya town | Madaya town | Kafr Al-Zair village | Kafr Al-Zair village | Ghassan Abboud roundabout | Ghassan Abboud roundabout | Al-Sena’a roundabout | Al-Sena’a roundabout`
* **Scores:** Spans Generated = 14 | Deduped BERTScore = 0.0000
---

**Edge Case #3 (Llama2.13B)**
* **Ground Truth:** `ISIL headquarters | civilian houses`
* **Model Prediction:** `school | school | School | school | market | market | coach station | swimming pool | health industry School`
* **Scores:** Spans Generated = 9 | Deduped BERTScore = 0.0000
---


#### D. Valid Paraphrasing (High Semantic Match, Zero Exact Match)
* **Total cases:** 5
* **By model:** Gpt Oss.20B: 2, Gemma3.4B: 1, Mistral: 1, Vicuna.13B: 1

**Edge Case #1 (Gpt Oss.20B)**
* **Ground Truth:** `Thermal Power plant`
* **Model Prediction:** `Thermal Power plant area`
* **Scores:** Span F1 = 0.0000 | Deduped BERTScore = 0.8119
---

**Edge Case #2 (Gemma3.4B)**
* **Ground Truth:** `militant hideout`
* **Model Prediction:** `militants' hideout`
* **Scores:** Span F1 = 0.0000 | Deduped BERTScore = 0.7191
---

**Edge Case #3 (Mistral)**
* **Ground Truth:** `militant hideout`
* **Model Prediction:** `militants' hideout`
* **Scores:** Span F1 = 0.0000 | Deduped BERTScore = 0.7191
---


#### E. Label Mismatch (High Text Match, Wrong Category)
* **Total cases:** 47
* **By model:** Llama3.1.8B: 12, Gemma4.E4B: 10, Mistral Nemo.12B: 6, Gemma3.4B: 5, Llama2.13B: 4, Vicuna.13B: 4, Gpt Oss.20B: 3, Mistral: 2, Qwen3.8B: 1

**Edge Case #1 (Gemma3.4B)**
* **Extracted Text:** `car | car` == `car | car`
* **Target Label:** `['Transportation/Marketing'] | ['Transportation/Marketing']`
* **Predicted Label:** `['Other'] | ['Other']`
* **Scores:** Span F1 = 1.0000 | Labeled Span F1 = 0.0000
---

**Edge Case #2 (Gemma4.E4B)**
* **Extracted Text:** `wine store` == `wine store`
* **Target Label:** `['Transportation/Marketing']`
* **Predicted Label:** `['Other']`
* **Scores:** Span F1 = 1.0000 | Labeled Span F1 = 0.0000
---

**Edge Case #3 (Gpt Oss.20B)**
* **Extracted Text:** `wine store` == `wine store`
* **Target Label:** `['Transportation/Marketing']`
* **Predicted Label:** `['Other']`
* **Scores:** Span F1 = 1.0000 | Labeled Span F1 = 0.0000
---

