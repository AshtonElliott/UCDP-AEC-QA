# UCDP-AEC-QA
This is the documentation of additional QA done on the UCDP-AEC Dataset found here:
https://github.com/ltgoslo/ucdp-aec

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

For LLMs, we have a script that can be ran locally (alongside Ollama) to output to a new JSON file that holds the LLM's response in a similiar structure to the one found in the exported Label Studio Json Format. The new JSON File can be found where the script is located.

Within the script, you can change the source JSON file location, the LLM model, and the name newly created JSON file.

To run the script, locate the folder it is located in and use the cd command in the Command Prompt:
```bash
  cd C:\Users\atown\OneDrive\Documents\ConfliBERT 
```

Then, you run the script with the following command:
```bash
  Python LocalQA.py
```
