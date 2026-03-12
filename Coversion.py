import json
import os

def convert_jsonl_to_json(input_jsonl_file, output_json_folder):
    # Ensure the output folder exists
    os.makedirs(output_json_folder, exist_ok=True)
    
    # Determine the output JSON filename
    base_name = os.path.splitext(os.path.basename(input_jsonl_file))[0]
    output_json_file = os.path.join(output_json_folder, base_name + '.json')
    
    # Read the JSONL file and aggregate the data
    data = []
    with open(input_jsonl_file, 'r') as jsonl_file:
        for line_number, line in enumerate(jsonl_file, start=1):
            line = line.strip()
            line = line.replace("}" , ", ")
            if not line:  # Skip empty lines
                continue
            try:
                data.append(json.loads(line + ' "question": "What arms or methods of force are used?"}'))
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON on line {line_number}: {e}")
                continue
    
    # Write to the JSON file
    with open(output_json_file, 'w') as json_file:
        json.dump(data, json_file, indent=4)
    
    print(f"Converted {input_jsonl_file} to {output_json_file}")

# Example usage
input_jsonl_file = 'C:/Users/ashto/Documents/UCDP-AEC Dataset/train.jsonl'  # Change this to your actual input file path
output_json_folder = 'C:/Users/ashto/Documents/UCDP-AEC Dataset/JSON_Files'  # Change this to your desired output folder path

convert_jsonl_to_json(input_jsonl_file, output_json_folder)