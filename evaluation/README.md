# Evaluation Pipeline

**Context:** This folder serves as a temporary, isolated environment to develop and test a new evaluation metric (using Optimal Bipartite Matching). To safely test this math without disrupting the team's current workflow or risking data loss in the main project files, this directory temporarily duplicates the logic found in `Local_QA.py`. Once the final dataset is completed and the evaluation logic is locked in, these scripts will be consolidated back into the main project pipeline.

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

**Data Directory Configuration:**
Create a dedicated folder named `data/` at the root of the project directory to house your working datasets.
1. Download the raw JSON export from our Label Studio project and place it inside the new folder as `data/label_studio_export.json`.
2. All data inputs, filtered train files, and inference outputs live inside this directory context.

*Note: The entire `data/` folder and generated JSON matrix outputs are intentionally blocked via `.gitignore` to prevent repository bloat and data leaks on GitHub.*

---

## 2. Folder Structure

The workspace has been organized into a modular, production-ready directory layout separating reusable core engine assets, temporary automation helpers, and visual notebook playgrounds:

```text
evaluation/
├── .venv/                      # Managed by uv (Do not touch)
├── data/                       # 📂 Inputs/Outputs (Blocked by Git)
│   ├── label_studio_export.json
│   ├── train.json              
│   └── mistral_Results.json    
├── src/                        # 📂 Pure, permanent evaluation package
│   ├── __init__.py             # Identifies src as a Python package
│   ├── eval_metrics.py         # Algorithmic engine (Hungarian Algorithm + EM)
│   └── visualization.py        # Reusable 4-panel visual analytics suite
├── notebook_experiments/       # 📂 Visual playgrounds & analytics reporting
│   └── baseline_mistral_eval.ipynb
├── temp_utility_scripts/       # 📂 Throwaway data generation & parsing tools
│   ├── label_studio_parser.py  
│   └── llm_inference.py        
├── .gitignore                  
├── .python-version
├── pyproject.toml
├── uv.lock
├── README.md
└── run_eval.py                 # 🚀 Main root-level pipeline execution script
```

---

## 3. Running the Pipeline

All scripts have been hardened with dynamic `os.path` path anchor matching. They calculate locations relative to their own source files, making them safe to run seamlessly from any working directory context. Execute the scripts in this exact sequence:

### Step 1: Isolate & Parse Human Annotations
Extracts and flattens labels belonging strictly to the target validator email address.
```bash
uv run temp_utility_scripts/label_studio_parser.py
```
*(Generates: `data/train.json`)*

### Step 2: Generate LLM Baseline Predictions
Runs the local inference loop over the filtered dataset using `ollama` to capture model span assertions.
```bash
uv run temp_utility_scripts/llm_inference.py
```
*(Generates: `data/mistral_Results.json`)*

### Step 3: Run the Main Production Metrics Dashboard
Executes the metrics engine to report system accuracy straight to your terminal profile.
```bash
uv run run_eval.py
```

### Step 4: Launch the 4-Panel Analytics Dashboard
Open `notebook_experiments/baseline_mistral_eval.ipynb` inside your IDE, verify your kernel is set to the starred `★ evaluation` environment, and run the visualization cell:
```python
from src.visualization import generate_evaluation_dashboard

# Evaluates any target dataset path instantly and visually
generate_evaluation_dashboard('../data/mistral_Results.json')
```

---

## 4. Methodology: Under the Hood

### The Core Matching Engine
1. **Word-Token Index Mapping:** The engine maps spans to distinct word tokens rather than raw character strings. It scans the entire article text to discover individual word boundaries using an alphanumeric word pattern (`\b\w+\b`). Every found word is assigned a sequential index number. If a word falls entirely within the starting and ending character boundaries of a highlight, its index number is added to that span's unique token set.

2. **Token Overlap Scoring (Intersection over Union):** To calculate the quality of a match, the script computes the Intersection over Union (IoU) of the token sets. It takes the number of identical word indices shared by both the human highlight and model prediction (the intersection) and divides it by the total count of unique word indices combined across both spans (the union). The engine uses a baseline threshold of `0.5` (50% word-token overlap) to count an extraction as a valid match.

3. **Global Pairing Optimization (The Hungarian Algorithm):** When an article contains multiple human highlights and multiple model predictions, simple loops can cause sorting errors where a messy prediction locks up a valid target and disrupts the rest of the scoring. The pipeline routes an IoU matrix to an assignment solver (`scipy.optimize.linear_sum_assignment`). This algorithm evaluates every possible pairing combination simultaneously to find the specific arrangement that achieves the highest total overlap score across the text block.

### Metrics
* **Exact Match (EM) Tracking:** Monitors perfect word-boundary matches where the IoU score reaches a perfect 1.0. This measures how often the model perfectly duplicates the starting and ending boundaries set by the reviewer.

* **Global Metric Pooling (Micro-Averaging)** Collects every single paired target, omission, and extra prediction from the entire dataset into one unified global pool before calculating final Precision, Recall, and F1-scores. This prevents a few unusually long or dense documents from warping the system-level metrics.

* **4-Panel Diagnostic Visuals:** Automatically feeds the pooled evaluation data into a visual reporting suite to break down errors into four analytical views:
  * **Document Token IoU Distribution:** Shows whether the model's predictions are landing near the targets or missing them entirely.
  * **F1 Score vs. Match Strictness:** Graphs how the final F1 score scales as the required evaluation threshold tightens anywhere from a loose `0.1` up to a strict `1.0` exact match.
  * **Extraction Quality Breakdown:** Displays a direct count of perfect character matches, valid partial hits, and complete omission errors.
  * **Token Confusion Matrix:** Organizes data into a four-quadrant map showing true extractions, missed highlights, and groundless extra predictions to pinpoint where accuracy is lost.

---

## 5. Potential Future Improvements

* **Standardized Evaluation Frameworks:** Migrating the code framework to use official open-source evaluation libraries to run standardized word-level classification checks that align directly with academic publishing benchmarks.

* **Semantic Similarity Evaluation:** Syntax engines evaluate synonymous context expressions (e.g., human labels "air raids" vs. model extracts "airstrikes") as a flat `0.0` failure. Future optimization will add contextual evaluation layers:
  * **BERTScore:** Projects strings into dense vector spaces to compute semantic cosine similarity mappings.
  * **LLM-as-a-Judge:** Piping unmatched False Positives to an advanced core model (e.g., GPT-4) with a verification prompt to score semantic alignment manually.
