# UCDP-AEC-QA
This is the documentation of additional QA done on the UCDP-AEC Dataset found here:
https://github.com/ltgoslo/ucdp-aec

## JSON Conversion
The JSONL files above are the UCDP-AEC Dataset that was converted from their id form. The details and how-to can be found on the original repository.

We have two versions of the conversion script:
  1. Conversion.py
  2. Conversion&Max1000.py

The only difference is the number of converted articles (there are roughly 10000 articles in total).

When using the JSON Convertor, you must change the filepath to where the jsonl file is currently located. The same ruling applies to the output path as well.

Ex:
```bash
input_jsonl_file = 'C:/Users/ashto/Documents/UCDP-AEC Dataset/train.jsonl'  # Change this to your actual input file path
```
```bash
output_json_folder = 'C:/Users/ashto/Documents/UCDP-AEC Dataset/JSON_Files'  # Change this to your desired output folder path
```

Once in JSON format, you can annotate them however you please manually. For us, we utilized Label Studio:
https://labelstud.io/

---
## Manual Annotation
The manual Annoation can be done any way you prefer. For us, it was using Label Studio. Should you also choose to use Label Studio, it is worth noting that the JSON exported will be condensed to one line. To ease the viewing process, we will provide a script here that creates a new json that holds the exported json's data in the non-one line format. Here's how to use it:
- Change the source directory to your exported JSON file (Line 4):
```bash
  with open('C:/Users/atown/OneDrive/Documents/ConfliBERT/APSA_Temp.json', 'r', encoding='utf-8') as f:
```
- Provide the result directory and name to your new formatted JSON file (Line 8):
```bash
  with open('C:/Users/atown/OneDrive/Documents/ConfliBERT/Annotated_APSA.json', 'w') as f:
```

---
## Local LLM Annotation

For LLMs, we have a script that can be ran locally (using Ollama) to output to a new JSON file that holds the LLM's response in a similiar structure to the one found in the exported Label Studio Json Format. The new JSON File can be found where the script is located.

The Script will require a version of Python and the Ollama Python Package. We recommend Python 3.12 as this is what it's built on.
For the package, use pip to install it in your local terminal:
```bash
  pip install ollama
```

Within the script, you can change the source JSON file location, the LLM model, and the name newly created JSON file. It can be found at the following lines:
- Source JSON file location (Line 8):
```bash
  input_path = os.path.join(script_dir, 'train.json')
```

- New JSON file name (Line 10):
```bash
  output_path = os.path.join(script_dir, 'mistral_results.json')
```

- LLM Model (Line 28):
```bash
  model='mistral',
```

---

To run the script, locate the folder the script is located in and use the cd command in the Command Prompt:
```bash
  cd C:\Users\atown\OneDrive\Documents\ConfliBERT 
```

Then, you run the script with the following command:
```bash
  python Local_QA.py
```

---
## HPC LLM Annotation

For those that have HPC access, we have a script and instructions as to how deploy and run our tools. Before running, ensure your HPC has GPU allocation and can allow multiple terminals at once.

## Ollama CLI Installation and serving on Terminal 1

```bash
# Enter the project folder
cd <YOUR_PROJECT_DIR>

# Download the Ollama Linux archive
curl -fL https://ollama.com/download/ollama-linux-amd64.tar.zst -o ollama.tar.zst

# Create a user-local install folder and extract Ollama there
mkdir -p ~/.local
tar -I zstd -xf ollama.tar.zst -C ~/.local

# Load Ollama from your user install and store models in the project folder. 
# MAKE SURE THE MODEL WEIGHTS FILES ARE STORED IN A FOLDER THAT DOES NOT HAVE TOO MUCH LIMITS ON THE SIZE.
export PATH="$HOME/.local/bin:$PATH"
export OLLAMA_MODELS="<YOUR_PROJECT_DIR>/ollama_models"

# allocate for resources (This is dependent on your HPC).
srun -p h100 --mem=32GB --time=01:00:00 --pty /bin/bash

# Check Ollama and start the server
ollama --version
ollama serve
```

## Running the script on Terminal 2

```bash
# Connect to previously allocated resource
ssh <Node_Name>*

# Enter the project folder
cd <YOUR_PROJECT_DIR>

# Load Ollama from your user install and store models in the project folder
export PATH="$HOME/.local/bin:$PATH"
export OLLAMA_MODELS="<YOUR_PROJECT_DIR>/Ollama_Models"

# Create and activate a project Conda environment. This can also be done with Python, but Conda is recommended for ease of use.
conda create -p <YOUR_PROJECT_DIR>/Python_Env python=3.12
conda activate $PWD/Python_Env

# Install the Python packages used by the scripts
pip install --upgrade pip
pip install ollama tqdm

# Pull the model from the running Ollama server, using the smaller model for testing
ollama pull mistral**

# Run the sync script or the async script
python HPC_QA.py
python HPC_QA_Async.py
```

## Checking status in terminal 3

```bash
nvidia-smi
ps aux | grep ollama
ps aux | grep python
watch -n 2 nvidia-smi
```
OR:

```bash
ml jobstats
jobstats <JOB_ID>
```

## Notes
*<Node_Name> is the Node that is being used. For example, if I used squeue --me and got the following:
```bash
               JOBID PARTITION     NAME     USER ST       TIME  NODES NODELIST(REASON)
            218313      h100     bash aee23000  R       9:18      1 g-04-02
```
The <Node_Name> would be g-04-02.

**mistral is the model examplified here, but you can swap this out with any model on Ollama.
---
