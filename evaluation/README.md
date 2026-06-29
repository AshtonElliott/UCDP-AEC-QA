# Evaluation Pipeline

**Context:** This folder serves as a temporary, isolated environment to develop and test an evaluation metric pipeline (using SQuAD Exact Match and BERTScore). To safely test this math without disrupting the team's current workflow or risking data loss in the main project files, this directory houses the updated logic. Once the final dataset is completed and the evaluation logic is locked in, these scripts will be consolidated back into the main project pipeline.

---

## 1. Environment & Workspace Setup

To guarantee perfect reproducibility across different machines (Mac, Windows, Linux), this project uses `uv` for lightning-fast dependency and virtual environment management.

**Prerequisites:**
* Install `uv` globally on your system.

**Initialization:**
You do not need to manually create a virtual environment or run standard `pip install`. Simply navigate to this folder in your terminal and run:

```bash
uv sync
```
*Note: This command reads the `uv.lock` and `pyproject.toml` files, automatically builds the hidden `.venv` folder, and installs the exact production versions of all dependencies required for this evaluation phase.*

---

## 2. Folder Structure

The workspace has been organized into a modular, production-ready directory layout separating reusable core engine assets, temporary automation helpers, and visual notebook playgrounds:

```text
evaluation/
├── .venv/                      # Managed by uv (Do not touch)
├── data/                       # 📂 Inputs/Outputs 
│   ├── train.json
│   ├── gemma3.4b_results.json         
│   └── gemma4.e4b_results.json    
│   └── llama3.1_Results.json    
│   └── mistral_Results (1).json    
│   └── qwen3.8b_Results.json    
├── src/                        # 📂 Pure, permanent evaluation package
│   ├── __init__.py             # Identifies src as a Python package
│   ├── eval_metrics.py         # Algorithmic engine 
│   └── visualization.py        # Reusable 4-panel visual analytics suite
├── notebook_experiments/       # 📂 Visual playgrounds & analytics reporting
│   └── baseline_mistral_eval.ipynb
├── temp_utility_scripts/       # 📂 Throwaway data generation & parsing tools
│   ├── label_studio_parser.py  
├── .gitignore                  
├── .python-version
├── pyproject.toml
├── uv.lock
├── README.md
└── run_eval.py                 # 🚀 Main root-level pipeline execution script
```

---

## 3. Running the Pipeline

All scripts have been hardened with dynamic `os.path` path anchor matching. They calculate locations relative to their own source files, making them safe to run seamlessly from any working directory context. 

```bash
uv run run_eval.py
```

---

## 4. Methodology: Under the Hood

This pipeline discards raw token-overlap/Regex matching in favor of an Orthogonal Evaluation Profile. By utilizing two distinct, peer-reviewed algorithms, we independently measure a model's strict formatting compliance against its actual semantic comprehension of political text.

### Axis 1: Structural Precision (SQuAD Exact Match)
To evaluate extractive discipline, the engine utilizes the official Stanford Question Answering Dataset (SQuAD) Exact Match methodology (Rajpurkar et al., 2016).

  * **Text Normalization:** Before scoring, the SquadEvaluator aggressively normalizes both the human ground truth and the model's prediction. It lowercases all text, strips punctuation, normalizes whitespace, and removes articles ("a", "an", "the").
  
  * **The Benefit:** This guarantees that LLMs are not unfairly penalized for trivial grammatical or formatting differences, while strictly enforcing that the model accurately captured the precise entity boundaries without hallucinating extra facts or characters.
 

### Axis 2: Semantic Comprehension (BERTScore)
To evaluate true conceptual understanding, the pipeline utilizes BERTScore (Zhang et al., ICLR 2020), passing batched strings to a frozen roberta-large contextual encoder.

* **Token-Level Cosine Similarity:** Instead of relying on brittle string matching, BERTScore projects every token into a high-dimensional vector space and calculates the pairwise cosine similarity between the human's text and the model's text.

* **The Benefit:** This mathematically resolves "phrasing bias." If a human annotates "air raid" and the model predicts "airstrike," SQuAD fails it with a 0.0, but BERTScore successfully scores it as a highly correlated match (e.g., 0.92), preventing the penalization of models that successfully comprehend the data but choose to paraphrase.
---

## 5. Potential Future Improvements

* **Metric Meta-Evaluation (Data Split):** Implementing a strict 10/90 data split to run a Pearson correlation ($r$) test. By isolating 10% of the dataset to compare human LabelStudio annotations directly against BERTScore outputs, we can mathematically prove the objective reliability of this metric specifically on the UCDP-AEC political corpus.

* **Error Taxonomy Generation:** Utilizing the outputted metrics to isolate the bottom 5% of catastrophic model failures and categorizing them (e.g., numerical blindness, implicit causality failures) to highlight specific architectural weaknesses in the LLMs.

* **Prompt Ablation Studies:** Testing different prompt architectures (Zero-Shot, Few-Shot, Chain-of-Thought) against this pipeline to study the impact on the Exact Match vs. Semantic F1 gap.
