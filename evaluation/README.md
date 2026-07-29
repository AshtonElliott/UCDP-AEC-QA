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

*Note: This command reads the `uv.lock` and `pyproject.toml` files, automatically builds the hidden `.venv` folder, and installs the exact production versions of all dependencies (pandas, bert_score, scipy, pingouin, seaborn, torch) required for this evaluation phase.*

---

## 2. Folder Structure

The workspace is organized into a modular, production-ready layout:

    evaluation/
    ├── data/                       # 📂 Inputs/Outputs 
    │   ├── completed_grading/      # Human 1-5 evaluation sheets
    │   ├── evaluation_results/     # Final markdown reports and JSON artifacts
    │   ├── grading_templates/      # Blind pilot sampling matrices
    │   └── raw_inputs/             # Raw JSON inferences and Ground Truth files
    ├── scripts/                    # 📂 Core evaluation logic & analytics
    │   ├── blind_sampler.py        # Generates blind grading sheets
    │   ├── calculate_correlation.py# Runs Spearman correlations (Human vs. Metric)
    │   ├── calculate_iaa.py        # Calculates Krippendorff's Alpha & ICC
    │   ├── core.py                 # The mathematical evaluation engine (SQuAD EM, DeBERTa, Token IoU)
    │   ├── run_eval.py             # Generates the master pipeline logic and leaderboards
    │   └── visualization.py        # Generates scatter plots and heatmaps for the report
    ├── temp_utility_scripts/       # 📂 Throwaway data generation & parsing tools
    ├── assets/                     # 📂 Output folder for generated PNG charts
    ├── main.py                     # 🚀 Master pipeline execution script
    ├── pyproject.toml
    └── README.md

---

## 3. Data Preparation

Before running the evaluation pipeline, you must populate the `data/raw_inputs/` folder with both the AI predictions and the human ground truth data.

**Step 1: Download the LLM Inferences**
The AI-generated answers are stored externally due to file size. 
1. Navigate to our shared Box space.
2. Download the raw LLM inference JSON files (e.g., `gemma3.4b_results.json`, `llama3.1.8b_results.json`).
3. Place all of these files directly into the `data/raw_inputs/` directory.

**Step 2: Build the Ground Truth**
The pipeline requires a finalized "consensus" ground truth to score the models against.
1. Download the raw human annotation exports from Label Studio.
2. Run the consensus generation script to merge conflicting human annotations into a single source of truth. 
3. **Important:** You must use the `--q` flag to generate the ground truth for each specific question track so they save correctly as `ground_truth_q1.json` and `ground_truth_q2.json`.

```bash
uv run scripts/consensus.py --q 1
uv run scripts/consensus.py --q 2
```

*Make sure the final output files are moved into the `data/raw_inputs/` directory before proceeding.*

---

## 4. Running the Master Pipeline

Once your data is prepared, you can generate the full evaluation report and build the visual assets by executing the master script from the root `evaluation` directory:

```bash
uv run main.py --pipeline
```

**Add-On Modifier Flags:**
* `--q [1 or 2]` : Filters the pipeline to only evaluate a specific question track. **This flag cannot run alone.** You must pair it with a primary execution flag.
  * *Example:* `uv run main.py --pipeline --q 2`

---

## 5. Understanding the Evaluation Metrics

This pipeline evaluates Information Extraction models using standard SQuAD 2.0 Macro-Averaging rules. Metrics are categorized into **Lexical Match** and **Semantic Match**. 

### The SQuAD 2.0 Scoring Paradigm: HasAns, NoAns, and Overall Score
To thoroughly evaluate both extraction capabilities and hallucination resistance, the dataset is split and scored as follows:

* **HasAns (Answerable):** Evaluated strictly on documents that contain valid target entities. This isolates the model's pure ability to extract and classify text correctly from a dense context.
* **NoAns / Abstention (Unanswerable):** Evaluated on documents that do not contain the target entities. This tests the model's safety and gatekeeping. If a model correctly outputs an empty array (`[]`), it receives a perfect `1.0` across all metrics for that document. If it hallucinates any spans, it receives a `0.0`.
* **Overall Score:** The combined Macro-average across *all* documents (both HasAns and NoAns). **Note:** Because a correct abstention on a NoAns document grants a free `1.0`, a dataset track with a high volume of unanswerable articles can artificially inflate the Overall score. To measure a model's true extraction capability, always compare the isolated `HasAns` score.

### Lexical Metrics (Surface-Level Text Match)
These metrics rely on exact string matching or word-level overlap.
* **Label EM (Strict Exact Match):** Calculates a document-level Macro F1 based on exact, inseparable tuple matches (Extracted Text + Category Label). It strictly penalizes missing labels, wrong categories, and over-generated duplicate spans.
* **SQuAD EM (Strict Exact Match):** Based on the official SQuAD 2.0 Set Exact Match metric. A score of `1.0` is awarded only if the model's extracted text perfectly matches the ground truth (disregarding punctuation and articles).
* **SQuAD Token F1 (Relaxed Lexical):** Calculates token-level overlap. It rewards models that extract the correct core phrase even if they include an extra word (e.g., "a gun" vs "gun"). While more forgiving than Exact Match, it still strictly requires exact word token matches.

### Semantic Metrics (Latent Context Match)
Because LLMs naturally paraphrase, lexical metrics often falsely penalize correct reasoning. We use semantic approximations to capture true contextual understanding:
* **Standard BERTScore:** Uses the `microsoft/deberta-large-mnli` model to calculate cross-contextual semantic similarity, scoring every predicted span against the ground truth to catch valid paraphrasing (e.g., "air raid" vs "airstrike").
* **Deduped BERTScore:** Removes redundant spans generated within the same document before passing them through DeBERTa, preventing models from artificially inflating their semantic scores via extreme verbosity.