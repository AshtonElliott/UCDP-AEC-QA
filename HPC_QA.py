import json
import os
import ollama
import re
import sys
 
# Unset proxies
os.environ.pop("http_proxy", None)
os.environ.pop("https_proxy", None)
 
# Load dataset
with open('/groups/pbrandt/aee230007/JSON_Files/train.json', 'r') as f:
    dataset = json.load(f)
 
if isinstance(dataset, dict):
    dataset = [dataset]
 
# Process entries
for idx, entry in enumerate(dataset):
    try:
        print(f"Processing entry {idx+1}/{len(dataset)}...")
        context = entry.get('source_article', '')
        question = entry.get('question', '')
        chunks = [context[i:i+1500] for i in range(0, len(context), 1500)]
        retrieved_context = "\n".join(chunks[:5])
        # Add timeout handling
        response = ollama.chat(
            model='gemma4:31b',
            messages=[
                {'role': 'system', 'content': 'Identify the words that answer the question. Return only a comma-separated list of words found in the article. There can be more than one answer to the question in the text.'},
                {'role': 'user', 'content': f"Context: {retrieved_context}\n\nQuestion: {question}"}   
            ]
        )
        prediction = response['message']['content']
        labels = prediction.split(',')
        spans = []
        for label in labels:
            clean_label = re.sub(r'[^\w\s]', '', label.strip())
            if clean_label:
                for match in re.finditer(re.escape(clean_label), context, re.IGNORECASE):
                    spans.append({
                        "end": match.end(),
                        "text": context[match.start():match.end()],
                        "start": match.start(),
                        "labels": ["Answer"]
                    })
        # Check AFTER processing all labels
        if len(spans) == 0:
            entry['no_answer'] = "No arms or methods mentioned"
        else:
            entry['answer_labels'] = spans
    except Exception as e:
        print(f"Error processing entry {idx}: {e}", file=sys.stderr)
        entry['error'] = str(e)
 
# Write ONCE after all entries processed
print("Writing results...")
with open('/groups/pbrandt/aee230007/Results/gemma_results.json', 'w') as f:
    json.dump(dataset, f, indent=4)
 
print("Process complete. Results saved.")