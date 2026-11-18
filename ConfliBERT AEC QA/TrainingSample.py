import json
import os

# n is the number of documents out of 1000.
def shorten_json(input_path, output_path, n=100):
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    subset = data[:n]

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(subset, f, ensure_ascii=False, indent=4)

script_dir = os.path.dirname(os.path.abspath(__file__))
input_json_fileq1 = os.path.join(script_dir, 'ground_truth_q1.json')
input_json_fileq2 = os.path.join(script_dir, 'ground_truth_q2.json')
output_json_folder = script_dir

shorten_json(input_json_fileq1, os.path.join(output_json_folder, 'ground_truth_q1_sample.json'))
shorten_json(input_json_fileq2, os.path.join(output_json_folder, 'ground_truth_q2_sample.json'))