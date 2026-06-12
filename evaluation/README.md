# Evaluation Pipeline



## 1. Environment Setup

To guarantee perfect reproducibility across different machines (Mac/Windows/Linux), I use [`uv`](https://docs.astral.sh/uv/) for dependency and virtual environment management.

**Prerequisites:**
* Install `uv` on your system.

**Initialization:**

You do not need to manually create a virtual environment or run `pip install`. Simply navigate to this folder in your terminal and run:

``` bash
uv sync
```

*Note: This command reads the `uv.lock` and `pyproject.toml` files, automatically builds the hidden `.venv` folder, and installs the exact versions of `numpy`, `scipy`, and `ollama` required for this evaluation phase.*

**Running the Pipeline**

Once the JSON results are generated, run the evaluation script locally.

``` bash
uv run evaluate.py
```

---

## 2. Methodology: Under the Hood

Standard NLP evaluation libraries (like `seqeval`) are built for rigid, token-by-token classification (BIO tagging) and will fail on our dataset because our text boundaries are fuzzy and contain massive amounts of duplicate keywords scattered across long documents. 

To solve this, our evaluation script utilizes **Optimal Bipartite Matching**.

### The Matching Engine
1. **Character-Level Sets:** Every text span is converted into a mathematical set of exact character indices (e.g., characters 120 through 135).
2. **Intersection over Union (IoU):** The script calculates the exact percentage overlap between the human's highlight and the AI's highlight. (Current Threshold: `0.5` or 50% overlap).
3. **The Hungarian Algorithm:** Instead of a simple "first-come, first-served" loop (which creates a greedy matching bias where a bad AI guess accidentally locks the wrong human span), we use `scipy.optimize.linear_sum_assignment`. This algorithm simultaneously evaluates a matrix of every possible combination of human and AI spans to find the mathematically perfect pairing that yields the highest total IoU score.

### Metrics Reported
* **Micro F1 / Precision / Recall:** Treats every span across the entire dataset as a single pool to measure global entity recognition accuracy.
* **Macro Document IoU:** Averages the score of each article independently to ensure the model's performance does not degrade on exceptionally long or short documents.

---

## 3. Folder Structure

* **`evaluate.py`**: The core evaluation engine (Hungarian Algorithm + IoU).
* **`pyproject.toml` & `uv.lock`**: Top-level dependencies and cryptographically locked sub-dependencies.
* **`.python-version`**: Forces the environment to use the exact correct Python execution engine.

---

## 4. Potential Future Improvements

* **Exact Match (EM) Tracking:** Currently, the pipeline grants partial credit for spans that meet the 0.5 IoU threshold. Adding an "Exact Match" metric (IoU = 1.0) alongside the F1 score will provide a stricter, binary view of how often the LLM perfectly duplicates human annotations.
* **Standardized SQuAD Evaluation:** Implementing the Hugging Face `evaluate` library (specifically the SQuAD metric). This evaluates at the token/word level rather than the character level and is widely recognized in academic NLP publishing. 
* **Semantic Evaluation:** Syntax matching fails if the AI predicts "airstrikes" but the human highlighted "air raids" (0.0 IoU). We may upgrade to evaluating *meaning* by implementing either:
  * **BERTScore:** Converts phrases into mathematical vectors to check for contextual similarity.
  * **LLM-as-a-Judge:** Routing unmatched False Positives to a stronger model (e.g., GPT-4) to ask if the AI's prediction is a functionally accurate synonym of the human's ground truth.