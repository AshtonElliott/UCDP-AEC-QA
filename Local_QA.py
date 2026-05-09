import json
import ollama
import re


# Change json source file here
with open('C:/Users/atown/OneDrive/Documents/ConfliBERT/AEC Dataset/JSON_Files/sample.json', 'r') as f:
    dataset = json.load(f)

# Wrap in a list if the file contains only one object
if isinstance(dataset, dict): dataset = [dataset]

for entry in dataset:
    context = entry.get('source_article', '')
    question = entry.get('question', '')
    
    # Change Model Here
    response = ollama.chat(model='qwen3.6:latest', messages=[
        {'role': 'system', 'content': 'Answer in exactly one or two words. No punctuation.'},
        {'role': 'user', 'content': f"Context: {context}\n\nQuestion: {question}"}
    ], options={'num_predict': 5})

    prediction = response['message']['content'].strip()
    clean_prediction = re.sub(r'[^\w\s]', '', prediction)
    
    spans = []
    if clean_prediction:
        for match in re.finditer(re.escape(clean_prediction), context, re.IGNORECASE):
            spans.append({
                "start":  match.start(),
                "text":  match.end(),
                "end": context[match.start():match.end()],
                "labels": ["Answer"]
            })
            
        if spans == []:
            entry['no_answer'] = "No arms or methods mentioned"
        else:
            entry['answer_labels'] = spans
            
    else:
        entry['no_answer'] = "No arms or methods mentioned"
    
    # Print Results to JSON; change JSON File Name and Filepath here
    with open('C:/Users/atown/OneDrive/Documents/ConfliBERT/Local Results/Temp.json', 'w') as f:
        json.dump(dataset, f, indent=4)

    print("Process complete. Results saved to newly created JSON File.")

    