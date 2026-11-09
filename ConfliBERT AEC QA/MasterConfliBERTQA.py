import subprocess
import sys
import os
import shutil
import argparse

script_dir = os.path.dirname(os.path.abspath(__file__))
eval_project_dir = os.path.abspath(os.path.join(script_dir, "..", "evaluation"))

parser = argparse.ArgumentParser(description="Master Runner: ConfliBERT QA Pipeline")
parser.add_argument("--model", type=str, default="all", help="Model to run ('all' or specific model)")
parser.add_argument("--tau", type=float, default=None, help="Null threshold override")
parser.add_argument("--alpha", type=float, default=None, help="Answer confidence ratio override")
parser.add_argument("--cache-raw", action="store_true", help="Cache raw candidate pools")
parser.add_argument("--skip-eval", action="store_true", help="Skip running evaluation pipeline at the end")
args = parser.parse_args()

qa_cmd = [sys.executable, 'appQA.py', '--model', args.model]
qa2_cmd = [sys.executable, 'appQAQ2.py', '--model', args.model]
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
subprocess.run(qa_cmd, cwd=script_dir, check=True)

# Step 2: Question 2
print("\n=== Running ConfliBERT QA (Question 2) ===")
subprocess.run(qa2_cmd, cwd=script_dir, check=True)

# Step 3: Run Eval Pipeline (if not caching and not skipped)
if not args.cache_raw and not args.skip_eval:
    print("\n=== Running Evaluation Pipeline ===")
    uv = shutil.which('uv') or os.path.join(os.path.expanduser('~'), '.local', 'bin', 'uv')
    if os.name == 'nt' and not uv:
        uv = os.path.join(os.path.expanduser('~'), '.local', 'bin', 'uv.exe')
    if uv and (shutil.which(uv) or os.path.exists(uv)):
        subprocess.run([uv, '--directory', eval_project_dir, 'run', 'main.py', '--pipeline'], check=True)
    else:
        # Fallback to python runner
        subprocess.run([sys.executable, 'main.py', '--pipeline'], cwd=eval_project_dir, check=True)
