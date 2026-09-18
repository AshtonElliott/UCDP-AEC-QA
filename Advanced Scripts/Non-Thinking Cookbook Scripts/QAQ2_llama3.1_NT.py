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
input_path = os.path.join(script_dir, 'train2.json')
output_path = os.path.join(script_dir, 'llama3.1.8b_results2_cb_NT.json')

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
            question = entry.get('question2', '')
            chunks = [context[i:i+1500] for i in range(0, len(context), 1500)]
            retrieved_context = "\n".join(chunks[:5])
            response = await client.chat(
                model='llama3.1:8b',
                format='json',
                messages=[
                    {'role': 'system', 'content': (
                    'Extract words answering the question and classify them into these 8 categories: '
                    'Energy, Water, Transportation/Marketing, Energy/Water, Health, Agriculture/Fishing, Government/Rebel, Other.'
                    'The category should be Energy when the infrastructure is related to energy exploration, production, and distribution.'
                    'The category should be Water when the infrastructure is related to drinking water, purification, irrigation, wastewatertreatment, and sanitation'
                    'The category should be Transportation/Marketing when the infrastructure is related to the transportation marketing, andexchange of commodities'
                    'The category should be Energy/Water when the infrastructure is related to both producing energy and water (e.g. a dam where hydropower and irrigation functions cannot be separated)'
                    'The category should be Health when the infrastructure is related to public health, including but not limited to hospitals, clinics, ambulances'
                    'The category should be Agriculture/Fishing when the infrastructure is related to crop cultivation and harvesting, and infrastructure related to fisheries'
                    'The category should be Government/Rebel when the infrastructure is related to Government or Public Based Buildings such as Schools, Admin Buildings, and Military Bases'
                    'The category should be Other when the infrastructure is not related to any of the previous categories.'
                    'Respond in JSON format with a list called "extractions" containing objects with keys "word" and "category".'
                    'If there is no answer, return the word Losolnachtnuma in the "extractions" list.'
                    )},
                
                    # Example 1: Standard infrastructure
                    {'role': 'user', 'content': f"Context: Rebels bombed the local bridge and the central hospital.\n\nQuestion: {question}"},
                    {'role': 'assistant', 'content': 'bridge | Transportation/Marketing, hospital | Health'},
                    
                    # Example 2: Multipurpose infrastructure
                    {'role': 'user', 'content': f"Context: The hydroelectric dam was targeted in the raid.\n\nQuestion: {question}"},
                    {'role': 'assistant', 'content': 'hydroelectric dam | Energy/Water'},
                    
                    # Example 3: Using Safe Word
                    {'role': 'user', 'content': f"Context: The town was targeted in the raid.\n\nQuestion: {question}"},
                    {'role': 'assistant', 'content': 'Losolnachtnuma'},
                    
                    {'role': 'user', 'content': f"Context: {retrieved_context}\n\nQuestion: {question}"}   
                ],
                think= False,
                options = {
                    "temperature": 0
                }                
            )
            prediction = response['message']['content']
            
            # Read in JSON Produced by Llama3.1
            data = json.loads(prediction)
            
            # Boolean to catch abstaining answers
            safeword = False
            
            if isinstance(data, dict):
                extractions = data.get('extractions', [])
            elif isinstance(data, list):
                extractions = data
            else:
                extractions = []
            
            spans = []
            for label in extractions:
                text = ""
                QALabel = ""
                if isinstance(label, dict):
                    text = label.get('word', '')
                    if text == "Losolnachtnuma":
                        safeword = True
                    QALabel = label.get('category', 'Other')
                
                if "" in text:
                    # Filter out blanks-positives
                    continue
                else:
                    # Apply Text & Label
                    for match in re.finditer(re.escape(text), context, re.IGNORECASE):
                        spans.append({
                            "end": match.end(),
                            "text": context[match.start():match.end()],
                            "start": match.start(),
                            "labels": [QAlabel]
                        })
            if len(spans) == 0 or QALabel == "" or safeword == True:
                if safeword == True:
                    entry['no_answer'] = "No arms or methods mentioned (Geniune No Answer)"
                else:
                    entry['no_answer'] = "No arms or methods mentioned (Non-Geniune No Answer)"
            else:
                entry['answer_labels'] = spans
            entry.pop('extractions', None)
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
