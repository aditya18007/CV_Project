import json 
from PIL import Image
import torchvision.transforms as transforms
import tqdm 

dataset_types = ['test', 'train', 'val']
transform = transforms.Compose([
    transforms.PILToTensor()
])

def read_data(dataset_type:str):
    if (dataset_type not in dataset_types):
        print(f"Invalid Dataset type requested ({dataset_type}). Type must be one of {dataset_types}")
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
        del data_point['image']
        data[data_point['id']] = data_point
    return data

if __name__ == "__main__":
    data = read_data('train')
    print(len(data))