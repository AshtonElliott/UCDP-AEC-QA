import json
import os

def convert_jsonl_to_json(input_jsonl_file, output_json_folder):
    # Ensure the output folder exists
    os.makedirs(output_json_folder, exist_ok=True)
    
    # Determine the output JSON filename
    base_name = os.path.splitext(os.path.basename(input_jsonl_file))[0]
    output_json_fileQ1 = os.path.join(output_json_folder, base_name + '.json')
    output_json_fileQ2 = os.path.join(output_json_folder, base_name + '2.json')
    
    # Read the JSONL file and aggregate the data
    counter = 0
    data = []
    with open(input_jsonl_file, 'r') as jsonl_file:
        for line_number, line in enumerate(jsonl_file, start=1):
            line = line.strip()
            line = line.replace("}" , ", ")
            if not line:  # Skip empty lines
                continue
            try:
                data.append(json.loads(line + ' "question": "What arms or methods of force are used?"}'))
                counter = counter + 1
                if counter == 1000:
                    break
                
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON on line {line_number}: {e}")
                continue
    
    # Repeat for 2nd Question
    
    # Write to the JSON file
    with open(output_json_fileQ1, 'w') as json_file:
        json.dump(data, json_file, indent=4)
    
    print(f"Converted {input_jsonl_file} to {output_json_fileQ1}")
    
    # Read the JSONL file and aggregate the data
    counter = 0
    data = []
    with open(input_jsonl_file, 'r') as jsonl_file:
        for line_number, line in enumerate(jsonl_file, start=1):
            line = line.strip()
            line = line.replace("}" , ", ")
            if not line:  # Skip empty lines
                continue
            try:
                data.append(json.loads(line + ' "question2": "What infrastructure or facilities were reported damaged?"}'))
                counter = counter + 1
                if counter == 1000:
                    break
                
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON on line {line_number}: {e}")
                continue
    
    # Write to the JSON file
    with open(output_json_fileQ2, 'w') as json_file:
        json.dump(data, json_file, indent=4)
        
    print(f"Converted {input_jsonl_file} to {output_json_fileQ2}")


# Change JSONL source file here if needed
script_dir = os.path.dirname(os.path.abspath(__file__))
# JSONL source file
input_jsonl_file = os.path.join(script_dir, 'train.jsonl')
# JSON folder
output_json_folder = script_dir

convert_jsonl_to_json(input_jsonl_file, output_json_folder)
