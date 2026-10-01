# ConfliBERT-QA AEC
As BERT architectures aren't supported on Ollama, we will instead utilized the ConfliBERT GUI v3 on Huggingface found here:
https://huggingface.co/spaces/eventdata-utd/ConfliBERT-GUI-v3

The one found in this repo has had its QA modified specifically for our needs.

## Utilization of the Modified GUI
 In order to use the GUI, you must clone this github repository. This can be done with the following code:
 
```bash
  git clone https://github.com/AshtonElliott/UCDP-AEC-QA
```
Then, navigate to this section of the cloned repo.

In order to use the QA here, your document must be a CSV. If you're results are in JSON, you can use our in-house python script, CSVConversion.py, to convert your JSON to CSV. You will need to either move the conversion script to where your JSON files are located or vice versa.

