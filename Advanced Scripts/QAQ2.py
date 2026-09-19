import json
import os
import ollama
import re
import sys
from typing import List, Literal
from pydantic import BaseModel, Field, ValidationError
import asyncio
from tqdm.asyncio import tqdm

# Pydantic Classes
# Enforce the 8 specific categories using Literal
class ExtractionLabels(BaseModel):
    word: str 
    category: Literal[ 
    "Energy",
    "Water",
    "Transportation/Marketing",
    "Energy/Water",
    "Health",
    "Agriculture/Fishing",
    "Government/Rebel",
    "Other"
    ] 
    
class ExtractionResponse(BaseModel):
    extractions: List[ExtractionLabels]

# Unset proxies
os.environ.pop("http_proxy", None)
os.environ.pop("https_proxy", None)

script_dir = os.path.dirname(os.path.abspath(__file__))
input_path = os.path.join(script_dir, 'train2sample.json')
output_path = os.path.join(script_dir, 'Invalid.json')

# Load dataset
with open(input_path, 'r') as f:
    dataset = json.load(f)

if isinstance(dataset, dict):
    dataset = [dataset]

sem = asyncio.Semaphore(5) # max concurrency
client = ollama.AsyncClient()

# Creating a function for one single async action
async def process_entry(idx, entry, ThinkingDet, Model):
    async with sem:
        try:
            print(f"Processing entry {idx+1}/{len(dataset)}...")
            context = entry.get('source_article', '')
            question = entry.get('question2', '')
            chunks = [context[i:i+1500] for i in range(0, len(context), 1500)]
            retrieved_context = "\n".join(chunks[:5])
            response = await client.chat(
                model=Model,
                format= ExtractionResponse.model_json_schema(),
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
                    'Respond in JSON format with a list called "extractions" containing objects with keys "word" and "category".'
                    'If there is no answer, return the word Losolnachtnuma in the "extractions" list.'
                    )},
                
                    # Example 1: Standard infrastructure
                    {'role': 'user', 'content': f"Context: Rebels bombed the local bridge and the central hospital.\n\nQuestion: {question}"},
                    {'role': 'assistant', 'content': '{"extractions": [{"word": "bridge", "category": "Transportation/Marketing"}, {"word": "hospital", "category": "Health"}]}'},
                    
                    # Example 2: Multipurpose infrastructure
                    {'role': 'user', 'content': f"Context: The hydroelectric dam was targeted in the raid.\n\nQuestion: {question}"},
                    {'role': 'assistant', 'content': '{"extractions": [{"word": "hydroelectric dam", "category": "Energy/Water"}'},
                    
                    # Example 3: Using Safe Word
                    {'role': 'user', 'content': f"Context: The town was targeted in the raid.\n\nQuestion: {question}"},
                    {'role': 'assistant', 'content': '{"extractions": [{"word": "Losolnachtnuma", "category": "Other"}]}'},
                    
                    {'role': 'user', 'content': f"Context: {retrieved_context}\n\nQuestion: {question}"}   
                ],
                think= ThinkingDet,
                options = {
                    "temperature": 0
                }
            )
            prediction = response['message']['content']
            
            # Read in JSON with Pydantic
            data = ExtractionResponse.model_validate_json(prediction)
                
            # Boolean to catch abstaining answers
            safeword = False
            
            spans = []
            for label in data.extractions:
                text = label.word.strip()
                if text == "Losolnachtnuma":
                    safeword = True
                    break
                if text:
                    # Apply Text & Label
                    spans.append({
                        "end": "N/A",
                        "text": [text],
                        "start": "N/A",
                        "labels": [label.category]
                    })
                    
            if safeword == True:
                entry['no_answer'] = "No arms or methods mentioned (Geniune No Answer)"
            else:
                entry['answer_labels'] = spans
        except ValidationError as e:
            print(f"Pydantic Validation Error in entry {idx}: {e}", file=sys.stderr)
            entry['error'] = f"Invalid schema returned: {e}"
        except Exception as e:
            print(f"Error processing entry {idx}: {e}", file=sys.stderr)
            entry['error'] = str(e)

# Main function to use async later on. 
# Use await instead of for loop for asyncio.
async def main():
    process = input("Select which process to Run (by number) \n 1. Non-Thinking \n 2. Thinking \n")
    # Change Model Here
    Model = 'gpt-oss:20b'
    
    if process == "1":
        await tqdm.gather(*[
            process_entry(idx, entry, False, Model)
            for idx, entry in enumerate(dataset)
        ])
        output_path = os.path.join(script_dir, 'gpt-oss.20b_results2_NT.json')
        
    elif process == "2": 
        await tqdm.gather(*[
            process_entry(idx, entry, True, Model)
            for idx, entry in enumerate(dataset)
        ])
        output_path = os.path.join(script_dir, 'gpt-oss.20b_results2_T.json')
        
    else:
        print("Invalid Selection")
        
    # Write ONCE after all entries processed
    print("Writing results...")
    with open(output_path, 'w') as f:
        json.dump(dataset, f, indent=4)
    print("Process complete. Results saved.")

# run main function async
asyncio.run(main())
