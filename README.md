# UCDP-AEC-QA
This is the documentation of additional QA done on the UCDP-AEC Dataset found here:
https://github.com/ltgoslo/ucdp-aec

The JSONL files above are the UCDP-AEC Dataset that was converted from their id form. The details and how-to can be found on the original repository.

When using the JSON Convertor, it is where to note that the input must be the directory to where the jsonl file is as well as your ideal directory. These are changed in "Conversion.py".

Ex:
input_jsonl_file = 'C:/Users/ashto/Documents/UCDP-AEC Dataset/train.jsonl'  # Change this to your actual input file path
output_json_folder = 'C:/Users/ashto/Documents/UCDP-AEC Dataset/JSON_Files'  # Change this to your desired output folder path
