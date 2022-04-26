import json 
from PIL import Image
import torchvision.transforms as transforms
import tqdm

from src.CONST import DATASET_TYPES 
from src.one_hot_encoding import one_hot_encode

def read_data(dataset_type:str):
    if (dataset_type not in DATASET_TYPES):
        print(f"Invalid Dataset type requested ({DATASET_TYPES}). Type must be one of {DATASET_TYPES}")
        exit(-1)
    
    with open(f'Data/annotations/{dataset_type}.jsonl', 'r',encoding='utf8') as json_file:
        json_list = list(json_file)

    data = []
    
    for json_str in tqdm.tqdm(json_list):
        data_point = json.loads(json_str)
        image_path = f"Data/images/{data_point['image']}"
        img = Image.open(image_path)

        data_point['image_PIL'] = img
        data_point['image_path'] = f"Data/images/{data_point['image']}"
        #Prevent some bad access from generic name
        del data_point['image']
        
        data_point['encoded_labels'] = one_hot_encode(data_point['labels'])
        data.append(data_point)
    #['id', 'labels', 'text', 'image_PIL', 'image_path', 'encoded_labels']
    return data


if __name__ == "__main__":
    data = read_data('train')
    for i in data:
        print(data[i].keys())
        break
    data = read_data('test')
    data = read_data('val')