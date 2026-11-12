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

Also, you need the proper python/conda environment to run the script. To do so, first install torch with either to use your cpu or gpu.

For CPU, run this command:
```bash
 pip install torch
```
For GPU, run this command:
```bash
 pip install torch --index-url https://download.pytorch.org/whl/cu124
```
Then, navigate to the this folder on your cloned repo in your terminal and run the following:
```bash
 pip install -r requirements.txt
```
After that's finish, you now have a working environment for the script.

## Running ConfliBERT-QA
 To use the ConfliBERT-QA, simply call the master script from this repo.
 ```bash
 python MasterConfliBERTQA.py
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
