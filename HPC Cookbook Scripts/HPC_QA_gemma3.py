import json
import os
import ollama
import re
import sys
import asyncio
from tqdm.asyncio import tqdm

# Unset proxies
os.environ.pop("http_proxy", None)
os.environ.pop("https_proxy", None)

script_dir = os.path.dirname(os.path.abspath(__file__))
input_path = os.path.join(script_dir, 'train.json')
output_path = os.path.join(script_dir, 'gemma3.4b_results_cb.json')

# Load dataset
with open(input_path, 'r') as f:
    dataset = json.load(f)

if isinstance(dataset, dict):
    dataset = [dataset]

sem = asyncio.Semaphore(5) # max concurrency
client = ollama.AsyncClient()

# Creating a function for one single async action
async def process_entry(idx, entry):
    async with sem:
        try:
            print(f"Processing entry {idx+1}/{len(dataset)}...")
            context = entry.get('source_article', '')
            question = entry.get('question', '')
            chunks = [context[i:i+1500] for i in range(0, len(context), 1500)]
            retrieved_context = "\n".join(chunks[:5])
            response = await client.chat(
                model='gemma3:4b',
                messages=[
                    {'role': 'system', 'content': ('Act as a document intelligence assistant.')}, 
                    {'role': 'user', 'content': (
                        f"Context: {retrieved_context}\n\n"
                        f"Question: {question}\n\n"  
                        'Identify the words that answer the question. Return only a comma-separated list of words found in the article. There can be more than one answer to the question in the text.'
                    )},
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

# Main function to use async later on. 
# Use await instead of for loop for asyncio.
async def main():
    await tqdm.gather(*[
        process_entry(idx, entry)
        for idx, entry in enumerate(dataset)
    ])
    # Write ONCE after all entries processed
    print("Writing results...")
    with open(output_path, 'w') as f:
        json.dump(dataset, f, indent=4)
    print("Process complete. Results saved.")

# run main function async
asyncio.run(main())
