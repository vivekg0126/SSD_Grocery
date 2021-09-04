import os.path as osp
import os
from math import sqrt
from statistics import mean, stdev
import json

location = './All_objects.json'
with open(location, 'r') as f:
    data = json.load(f)

train_data = data['train']

scale = []
aspect_ratio = []
for img in train_data:
    boxes = img['boxes']
    for b in boxes:
        sc = sqrt(b[2] * b[3])
        asp = b[2] / b[3]
        scale.append(sc)
        aspect_ratio.append(asp)

smean = mean(scale)
asp_mean = mean(aspect_ratio)
print('Scale mean :', smean)
print('Scale stddev:', stdev(scale, smean))
print('Aspect ratio mean:', asp_mean)
print('Aspect ratio stddev', stdev(aspect_ratio, asp_mean))