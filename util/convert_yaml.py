import json
import yaml

# Read the JSON file
with open('datasets/yr-earnings-occupation.json', 'r') as json_file:
    data = json.load(json_file)

# Create new structure without codes
simplified_data = {
    'occupations': [
        {
            'description': occupation['description'],
            'median': occupation['median']
        }
        for occupation in data['occupations']
    ]
}

# Write to YAML file
with open('output.yaml', 'w') as yaml_file:
    yaml.dump(simplified_data, yaml_file, sort_keys=False, allow_unicode=True)
