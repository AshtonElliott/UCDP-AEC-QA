import json
import ollama
import re


# Change json source file here
with open('C:/Users/atown/OneDrive/Documents/ConfliBERT/AEC Dataset/JSON_Files/train.json', 'r') as f:
    dataset = json.load(f)

# Wrap in a list if the file contains only one object
if isinstance(dataset, dict): dataset = [dataset]

for entry in dataset:
    context = entry.get('source_article', '')
    question = entry.get('question', '')
    
    chunks = [context[i:i+1500] for i in range(0, len(context), 1500)]
    retrieved_context = "\n".join(chunks[:5])
    
    # Change Model Here
    response = ollama.chat(model='mistral-small3.2', messages=[
        {'role': 'system', 'content': 'Identify the words that answer the question. Return only a comma-separated list of words found in the article. It can only be a maximum of 3 words, and there can be more than one answer to the question in the text.'},
        {'role': 'user', 'content': f"Context: {retrieved_context}\n\nQuestion: {question}"}
    ])

    prediction = response['message']['content']
    labels = prediction.split(',')
    
    spans = []
    for label in labels:
        clean_label = re.sub(r'[^\w\s]', '', label.strip())
        if clean_label:
            for match in re.finditer(re.escape(clean_label), context, re.IGNORECASE):
                spans.append({
                    "end": match.end(),
                    "text":  context[match.start():match.end()],
                    "start":  match.start(),
                    "labels": ["Answer"]
                })
                
            if spans == []:
                entry['no_answer'] = "No arms or methods mentioned"
            else:
                entry['answer_labels'] = spans
                
        else:
            entry['no_answer'] = "No arms or methods mentioned"
    
    # Print Results to JSON; change JSON File Name and Filepath here
    with open('C:/Users/atown/OneDrive/Documents/ConfliBERT/Local Results/Mistral-small3.2_Results.json', 'w') as f:
        json.dump(dataset, f, indent=4)

    print("Process complete. Results saved to newly created JSON File.")

    
