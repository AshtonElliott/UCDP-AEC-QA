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
input_path = os.path.join(script_dir, 'train2.json')
output_path = os.path.join(script_dir, 'gemma3.4b_results2_cb_NT.json')

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
                model='gemma3:4b',
                format= ExtractionResponse.model_json_schema(),
                messages=[
                    {'role': 'system', 'content': ('Act as a document intelligence assistant.')}, 
                    
                    {'role': 'user', 'content': (
                        f"Context: {retrieved_context}\n\n"
                        f"Question: {question}\n\n"
                    
                        'Identify the words that answer the question.'
                        'With every word, associate one of the 8 categories: Energy, Water, Transportation/Marketing, Energy/Water, Health, Agriculture/Fishing, Government/Rebel, Other. \n'
                    
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
                    {'role': 'assistant', 'content': '{"extractions": [{"word": "Losolnachtnuma", "category": "Other"}]}'}
                ]
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
