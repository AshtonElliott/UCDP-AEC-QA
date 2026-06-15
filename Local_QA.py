import json
import ollama
import re


script_dir = os.path.dirname(os.path.abspath(__file__))
# Change JSON source file here
input_path = os.path.join(script_dir, 'train.json')
#change Output JSON File Name here
output_path = os.path.join(script_dir, 'mistral_results.json')

with open(input_path, 'r') as f:
    dataset = json.load(f)

# Wrap in a list if the file contains only one object
if isinstance(dataset, dict): dataset = [dataset]

# Process entries
for idx, entry in enumerate(dataset):
    try:
        print(f"Processing entry {idx+1}/{len(dataset)}...")
        context = entry.get('source_article', '')
        question = entry.get('question', '')
        chunks = [context[i:i+1500] for i in range(0, len(context), 1500)]
        retrieved_context = "\n".join(chunks[:5])
        response = ollama.chat(
            # Change Model Here
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
    
    # Print Results to JSON
    with open(output_path, 'w') as f:
        json.dump(dataset, f, indent=4)

    print("Process complete. Results saved to newly created JSON File.")

    
