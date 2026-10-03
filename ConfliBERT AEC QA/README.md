# ConfliBERT-QA AEC
As BERT architectures aren't supported on Ollama, we will instead utilized the ConfliBERT GUI v3 on Huggingface found here:
https://huggingface.co/spaces/eventdata-utd/ConfliBERT-GUI-v3

The one found in this repo has had its QA modified specifically for our needs.

## QA Preparation
 In order to use the GUI, you must clone this github repository. This can be done with the following code:
 
```bash
  git clone https://github.com/AshtonElliott/UCDP-AEC-QA
```
Then, navigate to this section of the cloned repo.

In order to use the QA here, your document must be a CSV. If you're results are in JSON, you can use our in-house python script, CSVConversion.py, to convert your JSON to CSV. You will need to either move the conversion script to where your JSON files are located or vice versa.

The script assumes you have two JSON files, train.json and train2.json. If you have more or differently named scripts, you will need to change the variable names at the following locations:

CSVConversion.py (Lines 7-10):
```bash
 # Change/Add Extra Question Headers here.
column_names_by_file = {
    'ConfliBERT_results2': {'question2': 'question'},
}
```

CSVConversion.py (Lines 14 & 15):
```bash
  # Change FileNames and Number here
names = ['ConfliBERT_results', 'ConfliBERT_results2']
```

ResultsConversion.py (Lines 8-11):
```bash
 # Change/Add Extra Question Headers here.
column_names_by_file = {
    'ConfliBERT_results2': {'question': 'question2'},
}
```

ResultsConversion.py (Lines 17 & 18):
```bash
 # Change FileNames and Number here
names = ['train', 'train2']
```
Once there changes are made (if you had to), run the CSVConversion script. You should now have the CSV scripts that you need.

Lastly, you need the proper python/conda environment to run the GUI. To do so, first install torch with either to use your cpu or gpu.

For CPU, run this command:
```bash
 pip install torch
```
For GPU, run this command:
```bash
 pip install torch --index-url https://download.pytorch.org/whl/cu124
```
Then, navigate to the this folder on your cloned repo and run the following:
```bash
 pip install -r requirements.txt
```
After that's finish, you now have a working environment for the GUI.

## Utilization of the Modified GUI
