import json 
from PIL import Image
import torchvision.transforms as transforms
import tqdm

from CONST import DATASET_TYPES 
from one_hot_encoding import one_hot_encode

transform = transforms.Compose([
    transforms.PILToTensor()
])



def read_data(dataset_type:str):
    if (dataset_type not in DATASET_TYPES):
        print(f"Invalid Dataset type requested ({DATASET_TYPES}). Type must be one of {DATASET_TYPES}")
        exit(-1)
    
    with open(f'Data/annotations/{dataset_type}.jsonl', 'r') as json_file:
        json_list = list(json_file)

    data = {}
    
    for json_str in tqdm.tqdm(json_list):
        data_point = json.loads(json_str)
        image_path = f"Data/images/{data_point['image']}"
        img = Image.open(image_path)
        img_tensor = transform(img)
        
        data_point['image_tensor'] = img_tensor
        data_point['image_PIL'] = img
        data_point['image_path'] = f"Data/images/{data_point['image']}"
        #Prevent some bad access from generic name
        del data_point['image']
        
        data_point['encoded_labels'] = one_hot_encode(data_point['labels'])
        data[data_point['id']] = data_point
    
    return data


if __name__ == "__main__":
    data = read_data('train')
    data = read_data('test')
    data = read_data('val')