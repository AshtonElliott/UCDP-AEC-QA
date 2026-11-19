import subprocess
import sys
import os
import shutil
import argparse

script_dir = os.path.dirname(os.path.abspath(__file__))
eval_project_dir = os.path.abspath(os.path.join(script_dir, "..", "evaluation"))

parser = argparse.ArgumentParser(description="Master Runner: ConfliBERT QA Pipeline")
parser.add_argument("--model", type=str, default="all", help="Model to run ('all' or specific model)")
parser.add_argument("--base-model", type=str, default="all", help="Model to run Finetuning ('all' or specific model)")
parser.add_argument("--tau", type=float, default=None, help="Null threshold override")
parser.add_argument("--alpha", type=float, default=None, help="Answer confidence ratio override")
parser.add_argument("--cache-raw", action="store_true", help="Cache raw candidate pools")
parser.add_argument("--skip-eval", action="store_true", help="Skip running evaluation pipeline at the end")
parser.add_argument("--finetuned-dir", type=str, default=None,
                        help="Run the fine-tuned models saved by finetuneQA.py in this folder "
                             "instead of the originals (--model then picks which base models)")
args = parser.parse_args()

print("Would you like to run the models standard or finetuned?\n")
Finetuning = int(input("0. Standard\n1. Finetuned\n"))

if Finetuning == 1:
    qa_f_cmd = [sys.executable, 'finetuneQA.py', '--question', '1', '--train-data', 'ground_truth_q1_sample.json', '--base-model', args.base_model]
    qaq2_f_cmd = [sys.executable, 'finetuneQA.py', '--question', '2', '--train-data', 'ground_truth_q2_sample.json', '--base-model', args.base_model]
    
    subprocess.run(qa_f_cmd,cwd=script_dir, check=True)
    subprocess.run(qaq2_f_cmd,cwd=script_dir, check=True)

qa_cmd = [sys.executable, 'appQA.py', '--model', args.model]
qa2_cmd = [sys.executable, 'appQAQ2.py', '--model', args.model]

qa_fr_cmd = [sys.executable, 'appQA.py',  '--finetuned-dir', 'finetuned_q1', '--model', args.model]
qa2_fr_cmd = [sys.executable, 'appQAQ2.py',  '--finetuned-dir', 'finetuned_q2', '--model', args.model]

if args.tau is not None:
    qa_cmd.extend(['--tau', str(args.tau)])
    qa2_cmd.extend(['--tau', str(args.tau)])
if args.alpha is not None:
    qa_cmd.extend(['--alpha', str(args.alpha)])
    qa2_cmd.extend(['--alpha', str(args.alpha)])
if args.cache_raw:
    qa_cmd.append('--cache-raw')
    qa2_cmd.append('--cache-raw')

# Step 1: Question 1
print("\n=== Running ConfliBERT QA (Question 1) ===")
if Finetuning == 1:
    subprocess.run(qa_fr_cmd, cwd=script_dir, check=True)
else:
    subprocess.run(qa_cmd, cwd=script_dir, check=True)

# Step 2: Question 2
print("\n=== Running ConfliBERT QA (Question 2) ===")
if Finetuning == 1:
    subprocess.run(qaq2_fr_cmd, cwd=script_dir, check=True)
else:
    subprocess.run(qa2_cmd, cwd=script_dir, check=True)

# Step 3: Run Eval Pipeline (if not caching and not skipped)
if not args.cache_raw and not args.skip_eval:
    print("\n=== Running Evaluation Pipeline ===")
    uv = shutil.which('uv') or os.path.join(os.path.expanduser('~'), '.local', 'bin', 'uv')
    if os.name == 'nt' and not uv:
        uv = os.path.join(os.path.expanduser('~'), '.local', 'bin', 'uv.exe')
    if uv and (shutil.which(uv) or os.path.exists(uv)):
        subprocess.run([uv, '--directory', eval_project_dir, 'run', 'main.py', '--pipeline', '--report'], check=True)
    else:
        # Fallback to python runner
        subprocess.run([sys.executable, 'main.py', '--pipeline', '--report'], cwd=eval_project_dir, check=True)
