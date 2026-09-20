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
# Enforce the format
class ExtractionQ1(BaseModel):
    word: str 
    category: "Answer"
    
class ExtractionResponse(BaseModel):
    extractions: List[ExtractionQ1]

# Unset proxies
os.environ.pop("http_proxy", None)
os.environ.pop("https_proxy", None)

script_dir = os.path.dirname(os.path.abspath(__file__))
input_path = os.path.join(script_dir, 'train.json')
output_path = os.path.join(script_dir, 'gpt-oss.20b_results_NT.json')

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
            question = entry.get('question', '')
            chunks = [context[i:i+1500] for i in range(0, len(context), 1500)]
            retrieved_context = "\n".join(chunks[:5])
            response = await client.chat(
                model=Model,
                format= ExtractionResponse.model_json_schema(),
                messages=[
                    {'role': 'system', 'content': (
                    'Extract the words that answer the question. '
                    'Respond in JSON format with a list called "extractions" containing objects with the key "word".'
                    'If there is no answer, return the word Losolnachtnuma in the "extractions" list.'
                    )},
                    
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
            
            if isinstance(data, dict):
                extractions = data.get('extractions', [])
            elif isinstance(data, list):
                extractions = data
            else:
                extractions = []
            
            spans = []
            for label in extractions:
                text = label.word.strip()
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
        output_path = os.path.join(script_dir, 'gpt-oss.20b_results_NT.json')
        
    elif process == "2": 
        await tqdm.gather(*[
            process_entry(idx, entry, True, Model)
            for idx, entry in enumerate(dataset)
        ])
        output_path = os.path.join(script_dir, 'gpt-oss.20b_results_T.json')
        
    else:
        print("Invalid Selection")
        
    # Write ONCE after all entries processed
    print("Writing results...")
    with open(output_path, 'w') as f:
        json.dump(dataset, f, indent=4)
    print("Process complete. Results saved.")

# run main function async
asyncio.run(main())
