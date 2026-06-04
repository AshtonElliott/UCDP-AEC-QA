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

## Local LLM Annotation

For LLMs, we have a script that can be ran locally (using Ollama) to output to a new JSON file that holds the LLM's response in a similiar structure to the one found in the exported Label Studio Json Format. The new JSON File can be found where the script is located.

Within the script, you can change the source JSON file location, the LLM model, and the name newly created JSON file. It can be found at the following lines:
- Source JSON file location (Line 7):
```bash
  with open('C:/Users/atown/OneDrive/Documents/ConfliBERT/AEC Dataset/JSON_Files/sample.json', 'r') as f:
```

- LLM Model (Line 21):
```bash
  response = ollama.chat(model='gemma4:31b', messages=[
```

- New JSON file name and Filepath (Line 50):
```bash
  with open('C:/Users/atown/OneDrive/Documents/ConfliBERT/Local Results/Results.json', 'w') as f:
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

For the more larger or resource-heavy models, we have concocted a series of scripts that run the local script above in an HPC Environment. It should be noted that this has only been tested and fixed on the UTDallas Juno HPC and may not translate perfectly to every other HPC.

For the HPC, we have three scripts:
  - a Bash script for setup
  - a Bash script that prepares the HPC
  - the Python script

Within each script, the path locations can be changed to your needs. It can be found in the following lines:
- Conda Environment Filepath Location in PrepEnvironment.sh (Lines 4 & 7):
```bash
  conda create -p $PWD/Python_Env python=3.12
```
```bash
  conda activate $PWD/Python_Env
```

- Ollama Model Storage Filepath Location in RunLLM.sh (Line 7):
```bash
  OLLAMA_MODELS=$PWD/Ollama_Models
```

- Conda Environment Filepath Location in RunLLM.sh (Line 10):
```bash
  conda activate $PWD/Python_Env
```
- Source JSON file location (Line 7):
```bash
  with open('C:/Users/atown/OneDrive/Documents/ConfliBERT/AEC Dataset/JSON_Files/sample.json', 'r') as f:
```

- LLM Model (Line 21):
```bash
  response = ollama.chat(model='gemma4:31b', messages=[
```

- New JSON file name and Filepath (Line 50):
```bash
  with open('C:/Users/atown/OneDrive/Documents/ConfliBERT/Local Results/Results.json', 'w') as f:
```
---

When first running the on a HPC, run the PrepEnvironment.sh script to create the needed python environment and its dependencies. This will only need to be done once as long as the data in your files are kept.
```bash
  PrepEnvironment.sh
```

After Setup, runt the RunLLM.sh script to load Ollama, start a local Ollama Server, and run the python script. It is important to note that this version uses modules instead of a container, so this script will only work on your HPC provided:
a) Your HPC uses modules
b) Your HPC provides a module of Ollama

```bash
  RunLLM.sh
```

---
