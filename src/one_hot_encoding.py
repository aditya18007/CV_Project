import numpy as np 
import tqdm 
import json 
from CONST import DATASET_TYPES, NUM_LABELS, ONE_HOT_ENCODING

def print_encoding_dict():
    Labels = []
    for dataset_type in DATASET_TYPES:
        with open(f'Data/annotations/{dataset_type}.jsonl', 'r') as json_file:
            json_list = list(json_file) 
         
        for json_str in tqdm.tqdm(json_list):
            data_point = json.loads(json_str)
            Labels = Labels + data_point['labels']
    unique = set(Labels)
    one_hot_encoding = {}
    for i,label in enumerate(unique):
        one_hot_encoding[label] = i
    print(one_hot_encoding)

def one_hot_encode(labels):
    encoding = np.zeros((NUM_LABELS,))
    for label in labels:
        i = ONE_HOT_ENCODING[label]
        encoding[i] = 1
    return encoding