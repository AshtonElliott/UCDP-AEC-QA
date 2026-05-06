import json
import ollama

# Change json source file here
with open('C:/Users/atown/OneDrive/Documents/ConfliBERT/AEC Dataset/JSON_Files/sample.json', 'r') as f:
    dataset = json.load(f)

# Wrap in a list if the file contains only one object
if isinstance(dataset, dict): dataset = [dataset]

for entry in dataset:
    context = entry.get('source_article')
    question = entry.get('question')
    
    # Change Model Here
    response = ollama.chat(model='gemma4:31b', messages=[
        {'role': 'user', 'content': f"Context: {context}\n\nQuestion: {question}"}
    ])

    prediction = response['message']['content']
    
    entry['llm_response'] = prediction
    
    # Print Results to JSON; change JSON File Name here
    with open('Results.json', 'w') as f:
        json.dump(dataset, f, indent=4)

    print("Process complete. Results saved to newly created JSON File.")

    