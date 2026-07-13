# LLM Evaluation Pipeline: Extraction vs. Semantic Classification

---

## 1. Environment & Workspace Setup

To guarantee perfect reproducibility across different machines, this project uses `uv` for ultra-fast dependency and virtual environment management.

**Prerequisites:**
* Install `uv` globally on your system.

**Initialization:**
You do not need to manually create a virtual environment or run standard `pip install`. Simply navigate to this folder in your terminal and run:

```bash
uv sync
```
*Note: This command reads the `uv.lock` and `pyproject.toml` files, automatically builds the hidden `.venv` folder, and installs the exact production versions of all dependencies (pandas, bert_score, scipy, pingouin, seaborn) required for this evaluation phase.*

---

## 2. Folder Structure

The workspace is organized into a modular, production-ready layout:

```text
evaluation/
├── data/                       # 📂 Inputs/Outputs 
│   ├── completed_grading/      # Human 1-5 evaluation sheets
│   ├── evaluation_results/     # Final markdown reports and JSON artifacts
│   ├── grading_templates/      # Blind pilot sampling matrices
│   └── raw_inputs/             # Raw JSON inferences from LLMs (Gemma, Llama, Mistral, Qwen)
├── scripts/                    # 📂 Core evaluation logic & analytics
│   ├── blind_sampler.py        # Generates blind grading sheets
│   ├── calculate_correlation.py# Runs Spearman correlations (Human vs. Metric)
│   ├── calculate_iaa.py        # Calculates Krippendorff's Alpha & ICC
│   ├── core.py                 # The mathematical evaluation engine (SQuAD EM, DeBERTa, Token IoU)
│   ├── run_eval.py             # Generates the master pipeline logic and leaderboards
│   ├── error_analysis.py       # Captures edge cases (Paraphrasing, Over-Extraction, Label Mismatches)
│   └── visualization.py        # Generates scatter plots and heatmaps for the report
├── temp_utility_scripts/       # 📂 Throwaway data generation & parsing tools
├── assets/                     # 📂 Output folder for generated PNG charts
├── main.py                     # 🚀 Master pipeline execution script
├── pyproject.toml
└── README.md
```

---

## 3. Running the Master Pipeline

To generate the full evaluation report, build the visual assets, calculate Inter-Annotator Agreement (IAA), and run correlations, execute the master script through the `uv` environment:

```bash
uv run scripts/main.py --all
```

**Primary Execution Flags:**
You can isolate specific phases of the pipeline by passing targeted arguments:
* `--pipeline` : Runs the core LLM evaluation (Generates the leaderboard, heatmaps, and error analysis).
* `--iaa` : Runs human agreement calculations.
* `--correlation` : Runs Spearman correlation checks against human baselines.

**Add-On Modifier Flags:**
* `--q [1 or 2]` : Filters the pipeline to only evaluate a specific question track. **This flag cannot run alone.** You must pair it with a primary execution flag (e.g., `uv run scripts/main.py --pipeline --q 2`).

---

## 4. Understanding the Evaluation Metrics

This pipeline evaluates models across two dimensions: **Strict Lexical Match** and **Relaxed Semantic Match**.

### Strict Lexical Metrics
* **Label EM (Exact Match):** Calculates Precision and Recall based on exact tuple matches (Extracted Text + Category Label). 
* **Lexical (EM):** Based on the SQuAD Exact Match metric. A score of `1.0` is awarded only if the model's text perfectly matches the ground truth, disregarding punctuation and articles.

### Relaxed Semantic Metrics
Because LLMs naturally paraphrase, strict evaluation often falsely penalizes correct answers. We use semantic approximations to capture true understanding:
* **Token IoU:** Calculates the Intersection over Union of the words used. It rewards models that extract the correct core phrase even if they include an extra word (e.g., "a gun" vs "gun").
* **Deduped F1 (DeBERTa):** Uses the `DeBERTa-large-mnli` model to calculate cross-contextual semantic similarity. This catches valid paraphrasing (e.g., "Air Attack" vs "air raid") and scores them highly. 

### Diagnostics & Error Analysis
The pipeline automatically hunts for common LLM failure modes:
* **Valid Paraphrasing:** High Semantic score but zero Lexical score.
* **Over-Extraction (Noise):** Artificially inflating recall by generating an excessive number of spans. 
* **Label Mismatch:** Correctly extracting the text, but failing to categorize it into the correct taxonomy bucket.