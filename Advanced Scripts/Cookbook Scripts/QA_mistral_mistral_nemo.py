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
output_path = os.path.join(script_dir, 'invalid.json')

# Load dataset
with open(input_path, 'r') as f:
    dataset = json.load(f)

if isinstance(dataset, dict):
    dataset = [dataset]

sem = asyncio.Semaphore(5) # max concurrency
client = ollama.AsyncClient()

# Creating a function for one single async action
async def process_entry(idx, entry, Model, temp):
    async with sem:
        try:
            print(f"Processing entry {idx+1}/{len(dataset)}...")
            context = entry.get('source_article', '')
            question = entry.get('question', '')
            chunks = [context[i:i+1500] for i in range(0, len(context), 1500)]
            retrieved_context = "\n".join(chunks[:5])
            response = await client.chat(
                model=Model,
                format= 'json',
                messages=[
                    {'role': 'system', 'content': (
                        ' f"""" '
                        'You are a Poltical Scientist that is doing Manual Annotations on Text.' 
                        
                        '# Instructions:'
                        'Identify the words that answer the question. Return only a comma-separated list of words found in the article. There can be more than one answer to the question in the text.'
                        'Respond in JSON format with a list called "extractions" containing objects with keys "word" and "category".'
                        'If there is no answer, return the word Losolnachtnuma in the "extractions" list.'
                    )},
                    {'role': 'user', 'content': f"Context: {retrieved_context}\n\nQuestion: {question}"}
                ],
                think= False,
                options = {
                    "temperature": temp
                }
            )
            prediction = response['message']['content']
            
            # Read in JSON with Pydantic
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
                
                if isinstance(label, dict):
                   text = label.get('word', '')
                elif isinstance(label, str):
                   text = label
                else: 
                   text = ''
                
                if text == "Losolnachtnuma":
                    safeword = True
                if text:
                    # Apply Text & Label
                    spans.append({
                        "end": "N/A",
                        "text": [text],
                        "start": "N/A",
                        "labels": ["Answer"]
                    })
                    
            if safeword == True or len(spans) == 0:
                entry['no_answer'] = "No arms or methods mentioned (Geniune No Answer)"
            else:
                entry['answer_labels'] = spans
        except Exception as e:
            print(f"Error processing entry {idx}: {e}", file=sys.stderr)
            entry['error'] = str(e)

# Main function to use async later on. 
# Use await instead of for loop for asyncio.
async def main():
    process = input("Select which process to Run (by number) \n 1. Mistral \n 2. Mistral_Nemo \n")
    if process == "1":
        Model = 'mistral:7b'
        temp = 0.7
        await tqdm.gather(*[
            process_entry(idx, entry, Model, temp)
            for idx, entry in enumerate(dataset)
        ])
        output_path = os.path.join(script_dir, 'mistral.7b_results_cb_NT.json')
            
    elif process == "2":
        Model = 'mistral-nemo:12b'
        temp = 0.3
        await tqdm.gather(*[
            process_entry(idx, entry, Model, temp)
            for idx, entry in enumerate(dataset)
            ])
        output_path = os.path.join(script_dir, 'mistral_nemo.12b_results_cb_NT.json')
            
    else:
        print("Invalid Selection")
        
    # Write ONCE after all entries processed
    print("Writing results...")
    with open(output_path, 'w') as f:
        json.dump(dataset, f, indent=4)
    print("Process complete. Results saved.")

# run main function async
asyncio.run(main())
