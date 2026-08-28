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
output_path = os.path.join(script_dir, 'mistral_results2.json')

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
                model='mistral:latest',
                messages=[
                    {'role': 'system', 'content': (
                    'Identify the words that answer the question. Return only a comma-separated list of words found in the article.'
                    'With every word, associate one of the 8 categories below in the format: "Word | Category"'
                    'There can be more than one answer to the question in the text.'
                    'The 8 Categories are: Energy, Water, Transportation/Marketing, Energy/Water, Health, Agriculture/Fishing, Government/Rebel, Other.'
                    'The category should be Energy when the infrastructure is related to energy exploration, production, and distribution.'
                    'The category should be Water when the infrastructure is related to drinking water, purification, irrigation, wastewatertreatment, and sanitation'
                    'The category should be Transportation/Marketing when the infrastructure is related to the transportation marketing, andexchange of commodities'
                    'The category should be Energy/Water when the infrastructure is related to both producing energy and water (e.g. a dam where hydropower and irrigation functions cannot be separated)'
                    'The category should be Health when the infrastructure is related to public health, including but not limited to hospitals, clinics, ambulances'
                    'The category should be Agriculture/Fishing when the infrastructure is related to crop cultivation and harvesting, and infrastructure related to fisheries'
                    'The category should be Government/Rebel when the infrastructure is related to Government or Public Based Buildings such as Schools, Admin Buildings, and Military Bases'
                    'The category should be Other when the infrastructure is not related to any of the previous categories.'
                    'For every identified item, return only: "Text | Category". '
                    )},
                
                    # Example 1: Standard infrastructure
                    {'role': 'user', 'content': f"Context: Rebels bombed the local bridge and the central hospital.\n\nQuestion: {question}"},
                    {'role': 'assistant', 'content': 'bridge | Transportation/Marketing, hospital | Health'},
                    
                    # Example 2: Multipurpose infrastructure
                    {'role': 'user', 'content': f"Context: The hydroelectric dam was targeted in the raid.\n\nQuestion: {question}"},
                    {'role': 'assistant', 'content': 'hydroelectric dam | Energy/Water'},
                    
                    {'role': 'user', 'content': f"Context: {retrieved_context}\n\nQuestion: {question}"}   
                ],
                options = {
                    "temperature": 0
                }
            )
            prediction = response['message']['content']
            labels = prediction.split(',')
            spans = []
            for label in labels:
                clean_label = label.strip(' ".\' ')
                if '|' in clean_label:
                    # Separate Text and Label
                    parts = clean_label.split('|', 1)
                    text = parts[0]
                    
                    # Filter Out for Label
                    QAlabel = parts[1]
                        
                    if "Energy" in QAlabel:
                        QAlabel = "Energy"
                    elif "Water" in QAlabel:
                        QAlabel = "Water"
                    elif "Transportation/Marketing" in QAlabel:
                        QAlabel = "Transportation/Marketing"
                    elif "Energy/Water" in QAlabel:
                        QAlabel = "Energy/Water"
                    elif "Health" in QAlabel:
                        QAlabel = "Health"
                    elif "Agriculture/Fishing" in QAlabel:
                        QAlabel = "Agriculture/Fishing"
                    elif "Government/Rebel" in QAlabel:
                        QAlabel = "Government/Rebel"
                    else:
                        QAlabel = "Other"
                    
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
            if len(spans) == 0:
                if prediction == "Losolnichttproblem":
                    entry['no_answer'] = "No arms or methods mentioned (Geniune No Answer)"
                else:
                    entry['no_answer'] = "No arms or methods mentioned (Non-Geniune No Answer)"
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
