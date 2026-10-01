import json
import csv
import os

# JSON field name -> CSV column header. The JSON files are left unchanged.
column_names = {'source_article': 'context'}

script_dir = os.path.dirname(os.path.abspath(__file__))

for name in ('train', 'train2'):
    input_path = os.path.join(script_dir, name + '.json')
    output_path = os.path.join(script_dir, name + '.csv')

    with open(input_path, 'r', encoding='utf-8') as json_file:
        data = json.load(json_file)

    fieldnames = list(data[0].keys())
    header = [column_names.get(field, field) for field in fieldnames]

    with open(output_path, 'w', newline='', encoding='utf-8-sig') as csv_file:
        csv.writer(csv_file).writerow(header)
        csv.DictWriter(csv_file, fieldnames=fieldnames).writerows(data)
