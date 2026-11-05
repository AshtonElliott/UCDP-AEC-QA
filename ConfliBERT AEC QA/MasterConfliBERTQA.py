import subprocess
import sys

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

# Eval Pipeline