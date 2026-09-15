# LLM Evaluation Report
**Total Models Evaluated:** 8
**Total Documents Processed:** 1860
**Full Evaluation Compute Time:** 310.71 seconds

## Part 1: Global Benchmark Leaderboard (SQuAD 2.0 Evaluation)
> *All text metrics formatted as (Overall / HasAns)*

| Model            | Method        |   Doc Count |   Abstention (NoAns) |   Missed Answer Rate | Span F1 (Overall / HasAns)   | Labeled Span F1 (Overall / HasAns)   | SQuAD Token F1 (Overall / HasAns)   | BERTScore (Overall / HasAns)   |
|:-----------------|:--------------|------------:|---------------------:|---------------------:|:-----------------------------|:-------------------------------------|:------------------------------------|:-------------------------------|
| Gpt Oss.20B      | Raw (T)       |        1860 |                 0.01 |                 0    | 0.19 / 0.38                  | 0.19 / 0.38                          | 0.25 / 0.49                         | 0.26 / 0.51                    |
|                  | Raw (NT)      |        1860 |                 0.64 |                 0.02 | 0.51 / 0.38                  | 0.51 / 0.38                          | 0.56 / 0.49                         | 0.57 / 0.51                    |
|                  | Cookbook (T)  |        1860 |                 0    |                 0    | 0.00 / 0.01                  | 0.00 / 0.01                          | 0.00 / 0.01                         | 0.01 / 0.01                    |
|                  | Cookbook (NT) |        1860 |                 0    |                 0    | 0.00 / 0.01                  | 0.00 / 0.01                          | 0.01 / 0.01                         | 0.01 / 0.02                    |
| Gemma4.E4B       | Raw (T)       |        1860 |                 0.01 |                 0    | 0.19 / 0.37                  | 0.19 / 0.37                          | 0.24 / 0.48                         | 0.25 / 0.50                    |
|                  | Raw (NT)      |        1860 |                 0.66 |                 0.04 | 0.53 / 0.39                  | 0.53 / 0.39                          | 0.58 / 0.51                         | 0.59 / 0.52                    |
|                  | Cookbook (T)  |        1860 |                 0    |                 0    | 0.18 / 0.36                  | 0.18 / 0.36                          | 0.24 / 0.48                         | 0.24 / 0.49                    |
|                  | Cookbook (NT) |        1860 |                 0    |                 0    | 0.19 / 0.39                  | 0.19 / 0.39                          | 0.25 / 0.50                         | 0.25 / 0.51                    |
| Mistral Nemo.12B | Raw (NT)      |        1860 |                 0.59 |                 0.06 | 0.48 / 0.37                  | 0.48 / 0.37                          | 0.52 / 0.46                         | 0.54 / 0.48                    |
|                  | Cookbook (NT) |        1860 |                 0.67 |                 0.08 | 0.52 / 0.38                  | 0.52 / 0.37                          | 0.58 / 0.49                         | 0.59 / 0.51                    |
| Llama3.1.8B      | Raw (NT)      |        1860 |                 0.31 |                 0    | 0.33 / 0.35                  | 0.33 / 0.35                          | 0.38 / 0.44                         | 0.39 / 0.46                    |
|                  | Cookbook (NT) |        1860 |                 0.02 |                 0    | 0.22 / 0.41                  | 0.21 / 0.40                          | 0.28 / 0.53                         | 0.29 / 0.55                    |
| Mistral          | Raw (NT)      |        1860 |                 0.53 |                 0.05 | 0.43 / 0.32                  | 0.43 / 0.32                          | 0.47 / 0.41                         | 0.48 / 0.43                    |
|                  | Cookbook (NT) |        1860 |                 0.38 |                 0.02 | 0.34 / 0.31                  | 0.34 / 0.31                          | 0.39 / 0.41                         | 0.40 / 0.42                    |
| Gemma3.4B        | Raw (NT)      |        1860 |                 0.01 |                 0    | 0.15 / 0.30                  | 0.15 / 0.30                          | 0.20 / 0.39                         | 0.21 / 0.41                    |
|                  | Cookbook (NT) |        1860 |                 0    |                 0    | 0.16 / 0.33                  | 0.16 / 0.33                          | 0.21 / 0.42                         | 0.22 / 0.44                    |
| Vicuna           | Raw (NT)      |        1860 |                 0.48 |                 0.07 | 0.33 / 0.18                  | 0.33 / 0.18                          | 0.36 / 0.23                         | 0.35 / 0.23                    |
|                  | Cookbook (NT) |        1860 |                 0.02 |                 0.01 | 0.03 / 0.04                  | 0.02 / 0.03                          | 0.04 / 0.07                         | 0.04 / 0.06                    |
| Nemotron3.33B    | Raw (T)       |        1860 |                 0    |                 0    | 0.00 / 0.00                  | 0.00 / 0.00                          | 0.00 / 0.00                         | 0.00 / 0.00                    |
|                  | Raw (NT)      |        1860 |                 0.66 |                 0.06 | 0.52 / 0.38                  | 0.52 / 0.38                          | 0.57 / 0.47                         | 0.57 / 0.49                    |
|                  | Cookbook (T)  |        1860 |                 0    |                 0    | 0.18 / 0.36                  | 0.18 / 0.36                          | 0.25 / 0.50                         | 0.25 / 0.50                    |
|                  | Cookbook (NT) |        1860 |                 0.01 |                 0    | 0.18 / 0.36                  | 0.18 / 0.36                          | 0.25 / 0.50                         | 0.25 / 0.50                    |

---

## Part 2: Task Complexity Breakdown

### Question 1 Leaderboard
| Model            | Method        |   Doc Count |   Abstention (NoAns) |   Missed Answer Rate | Span F1 (Overall / HasAns)   | Labeled Span F1 (Overall / HasAns)   | SQuAD Token F1 (Overall / HasAns)   | BERTScore (Overall / HasAns)   |
|:-----------------|:--------------|------------:|---------------------:|---------------------:|:-----------------------------|:-------------------------------------|:------------------------------------|:-------------------------------|
| Gpt Oss.20B      | Raw (T)       |         954 |                 0.06 |                    0 | 0.38 / 0.46                  | 0.38 / 0.46                          | 0.49 / 0.59                         | 0.51 / 0.62                    |
|                  | Raw (NT)      |         954 |                 0.03 |                    0 | 0.37 / 0.46                  | 0.37 / 0.46                          | 0.48 / 0.59                         | 0.50 / 0.62                    |
|                  | Cookbook (T)  |         954 |                 0    |                    0 | 0.01 / 0.01                  | 0.01 / 0.01                          | 0.01 / 0.01                         | 0.01 / 0.02                    |
|                  | Cookbook (NT) |         954 |                 0    |                    0 | 0.01 / 0.01                  | 0.01 / 0.01                          | 0.01 / 0.02                         | 0.02 / 0.02                    |
| Gemma4.E4B       | Raw (T)       |         954 |                 0.03 |                    0 | 0.37 / 0.45                  | 0.37 / 0.45                          | 0.48 / 0.59                         | 0.49 / 0.61                    |
|                  | Raw (NT)      |         954 |                 0.08 |                    0 | 0.40 / 0.48                  | 0.40 / 0.48                          | 0.51 / 0.62                         | 0.52 / 0.63                    |
|                  | Cookbook (T)  |         954 |                 0    |                    0 | 0.35 / 0.44                  | 0.35 / 0.44                          | 0.46 / 0.58                         | 0.47 / 0.59                    |
|                  | Cookbook (NT) |         954 |                 0    |                    0 | 0.37 / 0.47                  | 0.37 / 0.47                          | 0.49 / 0.61                         | 0.50 / 0.62                    |
| Mistral Nemo.12B | Raw (NT)      |         954 |                 0.04 |                    0 | 0.37 / 0.45                  | 0.37 / 0.45                          | 0.45 / 0.55                         | 0.48 / 0.58                    |
|                  | Cookbook (NT) |         954 |                 0.05 |                    0 | 0.35 / 0.43                  | 0.35 / 0.43                          | 0.45 / 0.55                         | 0.47 / 0.58                    |
| Llama3.1.8B      | Raw (NT)      |         954 |                 0    |                    0 | 0.34 / 0.42                  | 0.34 / 0.42                          | 0.43 / 0.54                         | 0.45 / 0.56                    |
|                  | Cookbook (NT) |         954 |                 0    |                    0 | 0.34 / 0.42                  | 0.34 / 0.42                          | 0.43 / 0.54                         | 0.45 / 0.56                    |
| Mistral          | Raw (NT)      |         954 |                 0    |                    0 | 0.31 / 0.39                  | 0.31 / 0.39                          | 0.40 / 0.50                         | 0.42 / 0.52                    |
|                  | Cookbook (NT) |         954 |                 0.02 |                    0 | 0.31 / 0.38                  | 0.31 / 0.38                          | 0.40 / 0.49                         | 0.41 / 0.51                    |
| Gemma3.4B        | Raw (NT)      |         954 |                 0    |                    0 | 0.29 / 0.36                  | 0.29 / 0.36                          | 0.38 / 0.48                         | 0.40 / 0.50                    |
|                  | Cookbook (NT) |         954 |                 0    |                    0 | 0.32 / 0.40                  | 0.32 / 0.40                          | 0.41 / 0.51                         | 0.43 / 0.53                    |
| Vicuna           | Raw (NT)      |         954 |                 0    |                    0 | 0.17 / 0.22                  | 0.17 / 0.22                          | 0.23 / 0.28                         | 0.22 / 0.28                    |
|                  | Cookbook (NT) |         954 |                 0    |                    0 | 0.02 / 0.03                  | 0.02 / 0.03                          | 0.04 / 0.05                         | 0.03 / 0.04                    |
| Nemotron3.33B    | Raw (T)       |         954 |                 0    |                    0 | 0.00 / 0.00                  | 0.00 / 0.00                          | 0.00 / 0.00                         | 0.00 / 0.00                    |
|                  | Raw (NT)      |         954 |                 0    |                    0 | 0.37 / 0.46                  | 0.37 / 0.46                          | 0.46 / 0.58                         | 0.47 / 0.59                    |
|                  | Cookbook (T)  |         954 |                 0    |                    0 | 0.35 / 0.43                  | 0.35 / 0.43                          | 0.48 / 0.60                         | 0.49 / 0.61                    |
|                  | Cookbook (NT) |         954 |                 0.04 |                    0 | 0.36 / 0.44                  | 0.36 / 0.44                          | 0.49 / 0.60                         | 0.50 / 0.61                    |

### Question 2 Leaderboard
| Model            | Method        |   Doc Count |   Abstention (NoAns) |   Missed Answer Rate | Span F1 (Overall / HasAns)   | Labeled Span F1 (Overall / HasAns)   | SQuAD Token F1 (Overall / HasAns)   | BERTScore (Overall / HasAns)   |
|:-----------------|:--------------|------------:|---------------------:|---------------------:|:-----------------------------|:-------------------------------------|:------------------------------------|:-------------------------------|
| Gemma3.4B        | Raw (NT)      |         906 |                 0.01 |                 0    | 0.01 / 0.00                  | 0.01 / 0.00                          | 0.01 / 0.00                         | 0.01 / 0.00                    |
|                  | Cookbook (NT) |         906 |                 0    |                 0    | 0.00 / 0.00                  | 0.00 / 0.00                          | 0.00 / 0.00                         | 0.00 / 0.00                    |
| Gemma4.E4B       | Raw (T)       |         906 |                 0    |                 0    | 0.00 / 0.00                  | 0.00 / 0.00                          | 0.00 / 0.00                         | 0.00 / 0.00                    |
|                  | Raw (NT)      |         906 |                 0.8  |                 0.23 | 0.66 / 0.00                  | 0.66 / 0.00                          | 0.66 / 0.00                         | 0.66 / 0.00                    |
|                  | Cookbook (T)  |         906 |                 0    |                 0    | 0.00 / 0.00                  | 0.00 / 0.00                          | 0.00 / 0.00                         | 0.00 / 0.00                    |
|                  | Cookbook (NT) |         906 |                 0    |                 0    | 0.00 / 0.00                  | 0.00 / 0.00                          | 0.00 / 0.00                         | 0.00 / 0.00                    |
| Gpt Oss.20B      | Raw (T)       |         906 |                 0    |                 0    | 0.00 / 0.00                  | 0.00 / 0.00                          | 0.00 / 0.00                         | 0.00 / 0.00                    |
|                  | Raw (NT)      |         906 |                 0.79 |                 0.09 | 0.65 / 0.00                  | 0.65 / 0.00                          | 0.65 / 0.00                         | 0.65 / 0.00                    |
|                  | Cookbook (T)  |         906 |                 0    |                 0    | 0.00 / 0.00                  | 0.00 / 0.00                          | 0.00 / 0.00                         | 0.00 / 0.00                    |
|                  | Cookbook (NT) |         906 |                 0    |                 0    | 0.00 / 0.00                  | 0.00 / 0.00                          | 0.00 / 0.00                         | 0.00 / 0.00                    |
| Llama3.1.8B      | Raw (NT)      |         906 |                 0.39 |                 0.02 | 0.32 / 0.00                  | 0.32 / 0.00                          | 0.32 / 0.00                         | 0.32 / 0.00                    |
|                  | Cookbook (NT) |         906 |                 0.03 |                 0    | 0.09 / 0.36                  | 0.07 / 0.29                          | 0.11 / 0.51                         | 0.12 / 0.52                    |
| Mistral          | Raw (NT)      |         906 |                 0.67 |                 0.3  | 0.55 / 0.00                  | 0.55 / 0.00                          | 0.55 / 0.00                         | 0.55 / 0.00                    |
|                  | Cookbook (NT) |         906 |                 0.47 |                 0.12 | 0.38 / 0.00                  | 0.38 / 0.00                          | 0.38 / 0.00                         | 0.38 / 0.00                    |
| Mistral Nemo.12B | Raw (NT)      |         906 |                 0.73 |                 0.34 | 0.60 / 0.00                  | 0.60 / 0.00                          | 0.60 / 0.00                         | 0.60 / 0.00                    |
|                  | Cookbook (NT) |         906 |                 0.83 |                 0.43 | 0.70 / 0.12                  | 0.69 / 0.09                          | 0.71 / 0.19                         | 0.71 / 0.19                    |
| Nemotron3.33B    | Raw (T)       |         906 |                 0    |                 0    | 0.00 / 0.00                  | 0.00 / 0.00                          | 0.00 / 0.00                         | 0.00 / 0.00                    |
|                  | Raw (NT)      |         906 |                 0.83 |                 0.32 | 0.68 / 0.00                  | 0.68 / 0.00                          | 0.68 / 0.00                         | 0.68 / 0.00                    |
|                  | Cookbook (T)  |         906 |                 0    |                 0    | 0.00 / 0.00                  | 0.00 / 0.00                          | 0.00 / 0.00                         | 0.00 / 0.00                    |
|                  | Cookbook (NT) |         906 |                 0    |                 0    | 0.00 / 0.00                  | 0.00 / 0.00                          | 0.00 / 0.00                         | 0.00 / 0.00                    |
| Vicuna           | Raw (NT)      |         906 |                 0.61 |                 0.37 | 0.50 / 0.00                  | 0.50 / 0.00                          | 0.50 / 0.00                         | 0.50 / 0.00                    |
|                  | Cookbook (NT) |         906 |                 0.03 |                 0.03 | 0.04 / 0.11                  | 0.02 / 0.01                          | 0.05 / 0.16                         | 0.05 / 0.16                    |

---

## Part 3: Graphical Diagnostics

Generating Figure 1: Prompt Engineering Impact...
> This dumbbell plot shows if using prompt engineering (Cookbook) actually improves a model's extraction accuracy compared to its raw baseline.

![Figure 1: Prompt Engineering Impact](./assets/fig1_prompt_engineering_impact.png)

Generating Figure 2: Hallucination and Verbosity...
> These two panels show whether models fail by hallucinating on empty texts, and how their verbosity directly links to those errors.

![Figure 2: Hallucination and Verbosity](./assets/fig2_hallucination_and_verbosity.png)

Generating Figure 3: Classification Accuracy Drop...
> This bar chart focuses on Question 2 to show the performance penalty when a model successfully extracts the correct text but assigns the wrong category.

![Figure 3: Classification Accuracy Drop](./assets/fig3_classification_accuracy_drop.png)

Generating Figure 4: Reasoning vs. Prompting Comparison...
> This faceted bar chart compares only the thinking-capable models to see whether adding reasoning tokens or prompt engineering provides the bigger boost.

![Figure 4: Reasoning vs. Prompting Comparison](./assets/fig4_reasoning_vs_prompting.png)

