# LLM Evaluation Pipeline: Strict Extraction vs. Semantic Approximation

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
│   ├── evaluation_results/     # Final keys and generated outputs
│   ├── grading_templates/      # Blind pilot sampling matrices
│   └── raw_inputs/             # Raw JSON inferences from LLMs (Gemma, Llama, Mistral, Qwen)
├── scripts/                    # 📂 Core evaluation logic & analytics
│   ├── blind_sampler.py        # Generates blind grading sheets
│   ├── calculate_correlation.py# Runs Spearman correlations (Human vs. Metric)
│   ├── calculate_iaa.py        # Calculates Krippendorff's Alpha & ICC
│   ├── core.py                 # The mathematical evaluation engine (SQuAD, BERTScore, IoU)
│   ├── run_eval.py             # Generates the primary LLM leaderboard
│   ├── semantic_diagnostic.py  # Error analysis for complex edge cases
│   └── visualization.py        # Generates charts for the final report
├── temp_utility_scripts/       # 📂 Throwaway data generation & parsing tools
├── main.py                     # 🚀 Master pipeline execution script
├── evaluation_report.md        # The compiled final analysis report
├── pyproject.toml
└── README.md
```

---

## 3. Running the Master Pipeline

To generate the full evaluation report, calculate Inter-Annotator Agreement (IAA), run correlations, and build the visual assets, execute the master script:

```bash
uv run main.py --all
```
*You can also run specific flags (e.g., `--correlation` or `--iaa`) to isolate specific pipeline stages.*

---

