import json
import csv
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
column_names = {'context': 'source_article'}

# CSV stores everything as text; these fields are integers in the source JSON.
integer_fields = {'id', 'deaths_side_a', 'deaths_side_b', 'deaths_civilian',
                  'deaths_unknown', 'deaths_low', 'deaths_high'}

for name in ('ConfliBERT_results', 'ConfliBERT_results2'):
    input_path = os.path.join(script_dir, name + '.csv')
    output_path = os.path.join(script_dir, name + '.json')

    if not os.path.exists(input_path):
        print('skipping ' + name + '.csv (not found)')
        continue

    with open(input_path, 'r', newline='', encoding='utf-8-sig') as csv_file:
        data = []
        for row in csv.DictReader(csv_file):
            record = {}
            for column, value in row.items():
                field = column_names.get(column, column)
                if field in integer_fields and value != '':
                    value = int(value)
                record[field] = value
            data.append(record)

    with open(output_path, 'w', encoding='utf-8') as json_file:
        json.dump(data, json_file, ensure_ascii=False, indent=2)

    print(name + '.json: ' + str(len(data)) + ' records')
