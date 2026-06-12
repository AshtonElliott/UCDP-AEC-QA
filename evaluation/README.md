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