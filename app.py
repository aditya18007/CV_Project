import streamlit as st 
from PIL import Image
import json 
import clip.clip as clip 
from src.networks.CLIP_distilBERT_unfreeze import CLIP_dBERT_unfreezed_layers_dataset,CLIP_dBERT_unfreezed_layers_Input_transformer, CLIP_dBERT_unfreezed_layers_Model
import torch 
import re 
from src.CONST import NUM_LABELS
import sys 
import numpy as np 
from src.CONST import REVERSE_ONE_HOT_ENCODING

def get_text():
    Data = {}
    Images = []
    for dataset_type in ['test', 'train', 'val']:
        with open(f'Data/annotations/{dataset_type}.jsonl', 'r',encoding='utf8') as json_file:
            json_list = list(json_file)
        for json_str in json_list:
            data_point = json.loads(json_str)
            Data[ data_point['image'] ] = data_point['text']
            Images.append( data_point['image'] )

    return Data, set(Images)

def pretty_string(y_pred):
    techniques = []
    y_pred = y_pred.squeeze()
    for i in range(22):
        if y_pred[i] == 1:
            techniques.append(REVERSE_ONE_HOT_ENCODING[i])
    return techniques

def clean_text(text):
  text = text.replace("\n", " ")
  text = re.sub(r'^https?:\/\/.*[\r\n]*', '', text, flags=re.MULTILINE)
  return text

def predict(image_name,model):
    image_path = f"Data/images/{image_name}"
    img = Image.open(image_path)
    text = clean_text( Data_text[image_name] )
    dummy_encoding = np.zeros((NUM_LABELS,))
    data_point = {}
    data_point['text'] = text 
    data_point['image_PIL'] = img 
    data_point['encoded_labels'] = dummy_encoding
    dataset = CLIP_dBERT_unfreezed_layers_dataset([data_point], clip_img_processor)
    loader = torch.utils.data.DataLoader(dataset, batch_size=1)
    input_transformer = CLIP_dBERT_unfreezed_layers_Input_transformer()
    for x,y in loader:
        x,_ = input_transformer.transform(x,y)
        _, y_pred = model(*x)
        return pretty_string(y_pred)
    print("SHOULD NOT REACH HERE")
    sys.exit(-1)

def get_labels(uploaded_file,model):
    name = uploaded_file.name
    if name not in Images_list:
        return "Please select from test data. OCR limitation"
        sys.exit(-1)
    return predict(name, model)

def get_model():
   
    model = CLIP_dBERT_unfreezed_layers_Model(CLIP_MODEL)
    model.load_state_dict( torch.load('Models/CLIP_distilBERT_unfreeze.model') )
    model = model.cuda()
    model.eval()
    print("Succesfully Loaded model")
    return model

def main():

    st.title("Detected Propaganda techniques in memes")
    model = get_model()
    file_path = st.file_uploader("Choose an image...", type="png")
    
    if file_path is not None:
        image = Image.open(file_path)
        st.image(image, caption='Uploaded Image.', use_column_width=True)
        st.write("")
        st.write("Detected Propaganda techniques:")
        labels = get_labels(file_path,model)
        st.write(str(labels))

if __name__ == '__main__':
    Data_text, Images_list = get_text()
    CLIP_MODEL,  clip_img_processor =clip.load('ViT-B/32', jit=False)
    main()