import argparse
import sys

from scripts.calculate_correlation import run_correlation_pipeline
from scripts.calculate_iaa import calculate_iaa
from scripts.semantic_diagnostic import run_comprehensive_evaluation
from scripts.run_eval import run_comparative_pipeline

class MarkdownLogger:
    def __init__(self, filename="evaluation_report.md"):
        self.terminal = sys.stdout
        self.log_file = filename
        
        with open(self.log_file, "w", encoding="utf-8") as f:
            f.write("\n\n# LLM Evaluation Report\n")

    def write(self, message):
        # write to the terminal
        self.terminal.write(message)
            
        # write to the markdown file
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(message)

    def flush(self):
        self.terminal.flush()

def main():
    parser = argparse.ArgumentParser(description="Master pipeline for LLM Evaluation")
    
    parser.add_argument('--correlation', action='store_true', help='Run calculate_correlation.py')
    parser.add_argument('--iaa', action='store_true', help='Run calculate_iaa.py')
    parser.add_argument('--semantic', action='store_true', help='Run semantic_diagnostic.py')
    parser.add_argument('--pipeline', action='store_true', help='Run run_eval.py')
    parser.add_argument('--all', action='store_true', help='Run all pipelines')

    args = parser.parse_args()

    if not any(vars(args).values()):
        parser.print_help()
        return

    # turn on the logger 
    sys.stdout = MarkdownLogger("evaluation_report.md")

    if args.iaa or args.all:
        print("\n## 1. Inter-Annotator Agreement (Data Quality)\n")
        calculate_iaa()

    if args.correlation or args.all:
        print("\n## 2. Metric-to-Human Correlation\n")
        run_correlation_pipeline()
        
    if args.pipeline or args.all:
        print("\n## 3. LLM Performance Leaderboard\n")
        run_comparative_pipeline()

    if args.semantic or args.all:
        print("\n## 4. Error Analysis & Top Disagreements\n")
        run_comprehensive_evaluation()
        
    # restore standard output
    sys.stdout = sys.stdout.terminal
    print(f"\n[Run complete. Output appended to evaluation_report.md]")

if __name__ == "__main__":
    main()