import argparse
import sys

from scripts.calculate_correlation import run_correlation_pipeline
from scripts.calculate_iaa import calculate_advanced_iaa
from scripts.semantic_diagnostic import run_comprehensive_evaluation
from scripts.run_eval import run_comparative_pipeline

class MarkdownLogger:
    def __init__(self, filename="evaluation_report.md"):
        self.terminal = sys.stdout
        self.log_file = filename
        
        # start a markdown code block 
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write("\n\n```text\n")

    def write(self, message):
        # write to the terminal for live progress
        self.terminal.write(message)
        
        noise_markers = ["\r", "Processing article", "processing", "✅"]
        if any(marker in message for marker in noise_markers):
            return
            
        # write everything else to the file
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

    if args.correlation or args.all:
        print("\n--- Starting Correlation Pipeline ---")
        run_correlation_pipeline()

    if args.iaa or args.all:
        print("\n--- Starting IAA Analysis ---")
        calculate_advanced_iaa()
        
    if args.semantic or args.all:
        print("\n--- Starting Comprehensive Evaluation ---")
        run_comprehensive_evaluation()

    if args.pipeline or args.all:
        print("\n--- Starting Comparative Pipeline ---")
        run_comparative_pipeline()
        
    # close the md block
    with open("evaluation_report.md", "a", encoding="utf-8") as f:
        f.write("\n```\n")
    
    # restore standard output
    sys.stdout = sys.stdout.terminal
    print(f"\n[✅ Run complete. Output appended to evaluation_report.md]")

if __name__ == "__main__":
    main()