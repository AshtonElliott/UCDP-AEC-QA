import subprocess
import sys

# Convert to CSV
subprocess.run([sys.executable, 'CSVConversion.py'], capture_output=False, text=True)

# QA Process
subprocess.run([sys.executable, 'appQA.py'], capture_output=False, text=True)
subprocess.run([sys.executable, 'appQAQ2.py'], capture_output=False, text=True)

# Revert to JSON
subprocess.run([sys.executable, 'ResultsConversion.py'], capture_output=False, text=True)

# Eval Pipeline