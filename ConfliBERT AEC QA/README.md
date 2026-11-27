# ConfliBERT-QA AEC
As BERT architectures aren't supported on Ollama, we will instead utilized the ConfliBERT architecture from the ConfliBERT GUI v3 on Huggingface found here:
https://huggingface.co/spaces/eventdata-utd/ConfliBERT-GUI-v3

The one found in this repo has had its QA modified specifically for our needs.

## QA Preparation
 In order to use the QA, you must clone this github repository. This can be done with the following code:
 
```bash
  git clone https://github.com/AshtonElliott/UCDP-AEC-QA
```
Then, navigate to this section of the cloned repo.

In order to use the QA here, your document must be in a json or csv format. If your files are in JSONL, we have JSONL to json conversion script in the main section of this repo.

The script assumes you have two JSON files, train.json and train2.json.

Use the shared environment defined in `../evaluation/pyproject.toml` and
`../evaluation/uv.lock` for training, prediction, and evaluation. Install uv once:

```powershell
python -m pip install uv
```

## Running ConfliBERT-QA

From the `ConfliBERT AEC QA` folder, train and run model 2 for both questions,
then evaluate all saved results:

```powershell
python -m uv run --project ..\evaluation python MasterConfliBERTQA.py --base-model 2 --model 2 --run-finetuning 1
```

uv selects and synchronizes the project environment once. Every child process
uses that same Python through `sys.executable`; the master does not launch uv
again. GPU support comes from the project's CUDA-enabled PyTorch dependency.

To run the original model 2 without fine-tuning, omit `--run-finetuning 1`.
To generate predictions with saved fine-tuned checkpoints without retraining:

```powershell
python -m uv run --project ..\evaluation python appQA.py --finetuned-dir finetuned_q1 --model 2
python -m uv run --project ..\evaluation python appQAQ2.py --finetuned-dir finetuned_q2 --model 2
```

To rerun evaluation only:

```powershell
python -m uv run --project ..\evaluation python ..\evaluation\main.py --pipeline --report
```

This script runs both the QA and Eval pipeline. It has a number of arguments that can be used. These include:
```bash
parser.add_argument("--model", type=str, default="all", help="Model to run ('all' or specific model)")
parser.add_argument("--tau", type=float, default=None, help="Null threshold override")
parser.add_argument("--alpha", type=float, default=None, help="Answer confidence ratio override")
parser.add_argument("--cache-raw", action="store_true", help="Cache raw candidate pools")
parser.add_argument("--skip-eval", action="store_true", help="Skip running evaluation pipeline at the end")
```
After running the script, you can find the eval results within /evaluation/evaluation_report.md. We also have graphs to further illustrate the results which can be found in /evaluation/assets.
