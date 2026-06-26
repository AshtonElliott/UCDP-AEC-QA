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
output_path = os.path.join(script_dir, 'gemma_results.json')

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
                model='gemma4:31b', # ollama pull gemma3:4b, gemma4:12b, gemma4:31b
                model='llama3.1',
                messages=[
                    {'role': 'system', 'content': (
                    'Identify ALL damaged infrastructure or facilities reported in the article. '
                    'For every identified item, return exactly: "Text | Category". '
                    'Separate multiple items with a comma. '
                    'Categories: Energy, Water, Transportation/Marketing, Energy/Water, Health, Agriculture/Fishing, GOVERNMENT/REBEL, OTHER.'
                    'The category should be Energy when the infrastructure is related to energy exploration, production, and distribution.'
                    'The category should be Water when the infrastructure is related to drinking water, purification, irrigation, wastewatertreatment, and sanitation'
                    'The category should be Transportation/Marketing when the infrastructure is related to the transportation marketing, andexchange of commodities'
                    'The category should be Energy/Water when the infrastructure is related to both producing energy and water (e.g. a dam where hydropower and irrigation functions cannot be separated)'
                    'The category should be Health when the infrastructure is related to public health, including but not limited to hospitals, clinics, ambulances'
                    'The category should be Agriculture/Fishing when the infrastructure is related to crop cultivation and harvesting, and infrastructure related to fisheries'
                    'The category should be GOVERNMENT/REBEL when the infrastructure is related to Government or Public Based Buildings such as Schools, Admin Buildings, and Military Bases'
                    'The category should be OTHER when the infrastructure is not related to any of the previous categories.'
                    )},
                
                    # Example 1: Standard infrastructure
                    {'role': 'user', 'content': f"Context: Rebels bombed the local bridge and the central hospital.\n\nQuestion: {question}"},
                    {'role': 'assistant', 'content': 'bridge | Transportation/Marketing, hospital | Health'},
                    
                    # Example 2: Multipurpose infrastructure
                    {'role': 'user', 'content': f"Context: The hydroelectric dam was targeted in the raid.\n\nQuestion: {question}"},
                    {'role': 'assistant', 'content': 'hydroelectric dam | Energy/Water'},
                    
                    {'role': 'user', 'content': f"Context: {retrieved_context}\n\nQuestion: {question}"}   
                ]
            )
            prediction = response['message']['content']
            labels = prediction.split(',')
            spans = []
            for label in labels:
                clean_label = label.strip(' ".\' ')
                if '|' in clean_label:
                    text, QAlabel = [x.strip() for x in clean_label.split('|')]
                    
                    # Filter Out Extra Notes for Label
                    if 'Note:' in QAlabel:
                        QAlabel, LabelReasoning = [x.strip() for x in QAlabel.split('Note:')]
                        
                    for match in re.finditer(re.escape(text), context, re.IGNORECASE):
                        spans.append({
                            "end": match.end(),
                            "text": context[match.start():match.end()],
                            "start": match.start(),
                            "labels": [QAlabel]
                        })
            # Check AFTER processing all labels
            if len(spans) == 0:
                entry['no_answer'] = "No Damage Detected"
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
