import json
import os
import ollama
import re
import sys
from typing import List, Literal, Union
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
    extractions: List[Union[ExtractionLabels,str]]

# Unset proxies
os.environ.pop("http_proxy", None)
os.environ.pop("https_proxy", None)

script_dir = os.path.dirname(os.path.abspath(__file__))
input_path = os.path.join(script_dir, 'train2.json')
output_path = os.path.join(script_dir, 'vicuna_results2_cb.json')

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
                model='vicuna:13b',
                format=ExtractionResponse.model_json_schema(),
                messages=[
                    {'role': 'system', 'content': (
                        'A chat between a curious user and an artificial intelligence assistant. '
                        'The assistant gives helpful, detailed, and polite answers to the user\'s questions.'
                    )},
                    
                    {'role': 'user', 'content': (
                        f'Identify the words that answer the question. Return only a comma-separated list '
                        f'of words found in the article. With every word, associate one of the 8 categories '
                        f'below in the format: "Word | Category". \n'
                        f'Categories: Energy, Water, Transportation/Marketing, Energy/Water, Health, Agriculture/Fishing, Government/Rebel, and Other'
                        f'Question: {question} \n'
                        f'Context: {retrieved_context} \n'
                        f'For every identified item, return only: "Text | Category".'
                    )}
                ]
            )
            prediction = response['message']['content'].strip()
            prediction = re.sub(r'\]\}\s*\}$', ']}', prediction)
            
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
                    
            if safeword == True or len(spans) == 0:
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
