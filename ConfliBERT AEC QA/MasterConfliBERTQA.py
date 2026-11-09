import subprocess
import sys
import os
import shutil

eval_project_dir = r"C:\Users\ashto\OneDrive\Documents\GitHub\UCDP-AEC-QA\evaluation"
eval_data_dir = os.path.join(eval_project_dir, "data", "ConfliBERT Results")
script_dir = os.path.dirname(os.path.abspath(__file__))

# Convert to CSV
subprocess.run([sys.executable, 'CSVConversion.py'], capture_output=True, text=True)
print("CSV Conversion Completed.")

# QA Process
print("\nWelcome to ConfliBERTQA (Question 1)\n")
subprocess.run([sys.executable, 'appQA.py'], capture_output=False, text=True)
print("\nWelcome to ConfliBERTQA (Question 2)\n")
subprocess.run([sys.executable, 'appQAQ2.py'], capture_output=False, text=True)

# Revert to JSON
subprocess.run([sys.executable, 'ResultsConversion.py'], capture_output=False, text=True)

print("\n\n")

# Move Files to Eval Data Pipeline
names = [
           'ConfliBERT-cont-uncased-squad2_results', 'ConfliBERT-cont-cased-squad2_results', 'ConfliBERT-scr-uncased-squad2_results',
           'ConfliBERT-scr-cased-squad2_results', 'BERT-base-uncased-squad2_results', 'BERT-base-cased-squad2_results',
           'ConfliBERT-cont-uncased-squad2_results2', 'ConfliBERT-cont-cased-squad2_results2', 'ConfliBERT-scr-uncased-squad2_results2',
           'ConfliBERT-scr-cased-squad2_results2', 'BERT-base-uncased-squad2_results2', 'BERT-base-cased-squad2_results2',
           'ConfliBERT_QA-squad1_results', 'ConfliBERT_QA-squad1_results2'
        ]

for name in names:
    input_path = os.path.join(script_dir, name + '.json')
    output_path = os.path.join(eval_data_dir, name + '.json')

    if not os.path.exists(input_path):
        print('skipping ' + name + '.json (not found)')
        continue

    shutil.move(input_path, output_path)

# Start Eval Pipeline
uv = shutil.which('uv') or os.path.join(os.path.expanduser('~'), '.local', 'bin', 'uv.exe')
subprocess.run([uv, '--directory', eval_project_dir, 'run', 'main.py', '--pipeline'], check=True)