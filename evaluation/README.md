# Evaluation Pipeline

**Context:** 

This folder serves as a temporary, isolated testing environment to develop and test a new evaluation metric (using Optimal Bipartite Matching). To safely test this math without disrupting the team's current workflow or risking data loss in the main project files, this directory temporarily duplicates the logic found in `Local_QA.py`. Once the final dataset is completed and the evaluation logic is locked in, these scripts will be consolidated back into the main project pipeline.

---

## 1. Environment Setup

To guarantee perfect reproducibility across different machines (Mac, Windows, Linux), this project uses [`uv`](https://docs.astral.sh/uv/) for lightning-fast dependency and virtual environment management.

**Prerequisites:**
* Install `uv` on your system.

**Initialization:**
You do not need to manually create a virtual environment or run `pip install`. Simply navigate to this folder in your terminal and run:

```bash
uv sync
```

*Note: This command reads the `uv.lock` and `pyproject.toml` files, automatically builds the hidden `.venv` folder, and installs the exact versions of `numpy`, `scipy`, and `ollama` required for this evaluation phase.*

**Data Setup Requirement:**
To run this pipeline, you will need the raw temporary dataset. 
1. Download the JSON export from our Label Studio project.
2. Save it directly into this folder and name it `label_studio_export.json`.

*Note: Data files and generated JSON outputs are intentionally `.gitignore`d to prevent repository bloat.*

---

## 2. Running the Pipeline

To run a full test of the pipeline from raw data to final F1 score, execute the three scripts in this exact order:

**Step 1: Clean the Human Data**
```bash
uv run label_studio_export.py
```
*(Outputs: `train.json`)*

**Step 2: Generate AI Predictions**
```bash
uv run llm_inference.py
```
*(Outputs: `mistral_Results.json`)*

**Step 3: Run the Matrix Evaluation**

Once the data is ready, run the evaluation script locally:

```bash
uv run evaluate.py
```

---

## 3. Methodology: Under the Hood

Standard NLP evaluation libraries (like `seqeval`) are built for rigid, token-by-token classification (BIO tagging) and will fail on our dataset because our text boundaries are fuzzy and contain massive amounts of duplicate keywords scattered across long documents. 

To solve this, our evaluation script utilizes **Optimal Bipartite Matching**.

### The Matching Engine
1. **Character-Level Sets:** Every text span is converted into a mathematical set of exact character indices (e.g., characters 120 through 135).
2. **Intersection over Union (IoU):** The script calculates the exact percentage overlap between the human's highlight and the AI's highlight. (Current Threshold: `0.5` or 50% overlap).
3. **The Hungarian Algorithm:** Instead of a simple "first-come, first-served" loop (which creates a greedy matching bias where a bad AI guess accidentally locks the wrong human span), we use `scipy.optimize.linear_sum_assignment`. This algorithm simultaneously evaluates a matrix of every possible combination of human and AI spans to find the mathematically perfect pairing that yields the highest total IoU score.

### Metrics Reported
* **Micro Precision, Recall, and F1:** Treats every span across the entire dataset as a single pool to measure global entity recognition accuracy.
* **Macro Document IoU:** Averages the score of each article independently to ensure the model's performance does not degrade on exceptionally long or short documents.

---

## 4. Folder Structure

* **`label_studio_export.py`**: Cleans the raw JSON export from Label Studio and isolates user annotations.
* **`llm_inference.py`**: Contains the core LLM inference loop (duplicated from the original Local_QA script) to safely append AI guesses.
* **`evaluate.py`**: The core evaluation engine (Hungarian Algorithm + IoU).
* **`train.json`**: The cleaned, isolated human ground-truth dataset.
* **`mistral_Results.json`**: The final dataset containing both human `answer_labels` and AI `model_spans` side-by-side.
* **`pyproject.toml` & `uv.lock`**: Top-level dependencies and cryptographically locked sub-dependencies.
* **`.python-version`**: Forces the environment to use the exact correct Python execution engine.

---

## 5. Potential Future Improvements

* **Exact Match (EM) Tracking:** Currently, the pipeline grants partial credit for spans that meet the 0.5 IoU threshold. Adding an "Exact Match" metric (IoU = 1.0) alongside the F1 score will provide a stricter, binary view of how often the LLM perfectly duplicates human annotations.

* **Standardized SQuAD Evaluation:** Implementing the Hugging Face `evaluate` library (specifically the SQuAD metric). This evaluates at the token/word level rather than the character level and is widely recognized in academic NLP publishing. 

* **Semantic Evaluation:** Syntax matching fails if the AI predicts "airstrikes" but the human highlighted "air raids" (resulting in a 0.0 IoU). We may upgrade to evaluating *meaning* by implementing either:
  * **BERTScore:** Converts phrases into mathematical vectors to check for contextual similarity.
  
  * **LLM-as-a-Judge:** Routing unmatched False Positives to a stronger model (e.g., GPT-4) to ask if the AI's prediction is a functionally accurate synonym of the human's ground truth.
