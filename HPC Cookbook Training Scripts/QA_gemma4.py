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
    category: Literal[
    "Answer"
    ]
    
class ExtractionResponse(BaseModel):
    extractions: List[ExtractionQ1]

# Unset proxies
os.environ.pop("http_proxy", None)
os.environ.pop("https_proxy", None)

script_dir = os.path.dirname(os.path.abspath(__file__))
input_path = os.path.join(script_dir, 'train.json')
output_path = os.path.join(script_dir, 'gemma4.e4b_results_cb.json')

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
                model='gemma4:e4b',
                format=ExtractionResponse.model_json_schema(),
                messages=[
                    {'role': 'system', 'content': (
                        '<|think|>'
                        'Identify the words that answer the question. Return only a comma-separated list of words found in the article. There can be more than one answer to the question in the text.'
                    )},
                    {'role': 'user', 'content': f"Context: {retrieved_context}\n\nQuestion: {question}"}
                ]
                
                options = {
                    'top_p': 0.95,
                    'top_k': 64
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
            
            # Read Each Article and Answer into a span
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
