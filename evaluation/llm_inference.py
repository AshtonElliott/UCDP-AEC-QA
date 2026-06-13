import json
import ollama
import re
import sys

with open('train.json', 'r') as f:
    dataset = json.load(f)

run_slice = dataset 

for idx, entry in enumerate(run_slice):
    try:
        print(f"Processing entry {idx+1}/{len(run_slice)}...")
        context = entry.get('source_article', '')
        question = entry.get('question', '')
        
        # chunk text to stay safely within context boundaries
        chunks = [context[i:i+1500] for i in range(0, len(context), 1500)]
        retrieved_context = "\n".join(chunks[:5])
        
        response = ollama.chat(
            model='mistral',
            messages=[
                {'role': 'system', 'content': (
                    'Identify a word or string of words that answer the question.'
                    'Return only a comma-separated list of specific, concrete answers found in the article.'
                )},
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
                        "start": match.start(),
                        "end": match.end(),
                        "text": context[match.start():match.end()],
                        "labels": ["Answer"]
                    })
                    
        # save to a distinct key 
        if len(spans) == 0:
            entry['model_no_answer'] = "No arms or methods mentioned"
        else:
            entry['model_spans'] = spans
            
    except Exception as e:
        print(f"Error processing entry {idx}: {e}", file=sys.stderr)
        entry['error'] = str(e)

# save the combined data (Human + AI) into a master results file
with open('mistral_Results.json', 'w') as f:
    json.dump(run_slice, f, indent=4, ensure_ascii=False)

print("Generation complete! Combined dataset saved to mistral_Results.json")