import argparse
import sys
from pathlib import Path

from scripts.calculate_correlation import run_correlation_pipeline
from scripts.calculate_iaa import calculate_iaa
from scripts.semantic_diagnostic import run_comprehensive_evaluation
from scripts.run_eval import run_evaluation_pipeline

BASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BASE_DIR / 'data' / 'evaluation_results'

class MarkdownLogger:
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log_file = Path(filename)
        
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        
        with self.log_file.open("w", encoding="utf-8") as f:
            f.write("# LLM Evaluation Report\n")

    def write(self, message):
        self.terminal.write(message)
        with self.log_file.open("a", encoding="utf-8") as f:
            f.write(message)

    def flush(self):
        self.terminal.flush()

def main():
    parser = argparse.ArgumentParser(description="Master pipeline for LLM Evaluation")
    
    parser.add_argument('--correlation', action='store_true', help='Run calculate_correlation.py')
    parser.add_argument('--iaa', action='store_true', help='Run calculate_iaa.py')
    parser.add_argument('--semantic', action='store_true', help='Run semantic_diagnostic.py')
    parser.add_argument('--pipeline', action='store_true', help='Run run_eval.py (includes error analysis)')
    parser.add_argument('--all', action='store_true', help='Run all pipelines')
    parser.add_argument('--q', type=int, default=None, help='Filter pipeline to a specific question (e.g. 1 or 2)')

    args = parser.parse_args()

    if not any([args.correlation, args.iaa, args.semantic, args.pipeline, args.all]):
        parser.print_help()
        return

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    report_path = BASE_DIR / "evaluation_report.md"
    
    sys.stdout = MarkdownLogger(report_path)

    if args.iaa or args.all:
        calculate_iaa()

    if args.correlation or args.all:
        run_correlation_pipeline()
        
    if args.pipeline or args.all:
        run_evaluation_pipeline(question_filter=args.q)

    if args.semantic or args.all:
        run_comprehensive_evaluation()

    sys.stdout = sys.stdout.terminal
    print(f"\n[Run complete. Output saved directly to {report_path}]")

if __name__ == "__main__":
    main()