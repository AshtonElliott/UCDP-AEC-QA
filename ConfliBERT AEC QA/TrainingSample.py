import json
import os
import random

# n is the number of documents out of 1000.
def shorten_json(input_path, output_path, n=100):
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    subset = data[:n]

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(subset, f, ensure_ascii=False, indent=4)

# Added function to scale number of actually answered Q2 question to better train ConfliBERT.        
def finetuned_shorten_json(input_path, output_path, n_answers=75, n_no_answers=25, seed=42):
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    with_answers = [doc for doc in data if doc.get('answer_labels')]
    without_answers = [doc for doc in data if not doc.get('answer_labels')]

    rng = random.Random(seed)
    subset = rng.sample(with_answers, n_answers) + rng.sample(without_answers, n_no_answers)
    rng.shuffle(subset)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(subset, f, ensure_ascii=False, indent=4)

    print(f"Wrote {len(subset)} documents ({n_answers} with answer_labels, "
          f"{n_no_answers} with no_answer) to {output_path}")

script_dir = os.path.dirname(os.path.abspath(__file__))
input_json_fileq1 = os.path.join(script_dir, 'ground_truth_q1.json')
input_json_fileq2 = os.path.join(script_dir, 'ground_truth_q2.json')
output_json_folder = script_dir

shorten_json(input_json_fileq1, os.path.join(output_json_folder, 'ground_truth_q1_sample.json'))
finetuned_shorten_json(input_json_fileq2, os.path.join(output_json_folder, 'ground_truth_q2_sample.json'))