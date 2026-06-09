import json

# Load NewTemplate.txt to extract the new ID
with open('C:/Users/atown/OneDrive/Documents/ConfliBERT/APSA_Temp.json', 'r', encoding='utf-8') as f:
    Annotated_data = json.load(f)

if isinstance(Annotated_data, dict): Annotated_data = [Annotated_data]

# Save the results to a new file
with open('C:/Users/atown/OneDrive/Documents/ConfliBERT/Annotated_APSA.json', 'w') as f:
    json.dump(Annotated_data, f, indent=4)
