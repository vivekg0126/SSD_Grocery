import json

file = 'All_objects.json'

with open(file, 'r') as f:
    x = json.load(f)

test_data = x['test']
img_2_prd = {}
for data in test_data:
    img_2_prd[data['name']] = data['objects']

with open('image2products.json', 'w') as g2p:
    json.dump(img_2_prd,g2p, indent=2)