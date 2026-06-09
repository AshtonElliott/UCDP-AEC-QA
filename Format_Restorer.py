import json

# Load results from exported JSON here
with open('C:/Users/atown/OneDrive/Documents/ConfliBERT/APSA_Temp.json', 'r', encoding='utf-8') as f:
    Annotated_data = json.load(f)

# Save the results to a new JSON
with open('C:/Users/atown/OneDrive/Documents/ConfliBERT/Annotated_APSA.json', 'w') as f:
    json.dump(Annotated_data, f, indent=4)
