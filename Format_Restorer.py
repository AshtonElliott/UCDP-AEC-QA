import json
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
unformatted_file = os.path.join(script_dir, 'Temp_APSA.json')
formatted_file = os.path.join(script_dir, 'Annotated_APSA.json')

with open(unformatted_file, 'r', encoding='utf-8') as f:
    Annotated_data = json.load(f)

# Save the results to a new JSON
with open(formatted_file, 'w') as f:
    json.dump(Annotated_data, f, indent=4)
