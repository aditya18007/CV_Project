import os 
import wget 
import torch 
import numpy as np 

CLIP = 'CLIP'

MODELS = [CLIP]
Links = {
    CLIP:'https://openaipublic.azureedge.net/clip/models/40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt'
}

def get_model(model_name:str):
    if model_name not in MODELS:
        print(f"Invalid Model name : {model_name}. Valid arguements = {MODELS}")
        exit(-1)

    if not os.path.exists('Models'):
        os.mkdir('Models')

    model_filepath = f'Models/{model_name}.pt'
    if not os.path.exists(model_filepath):
        wget.download(Links[model_name], out=model_filepath)
    if model_name == CLIP:
        if not torch.cuda.is_available():
            print("Cuda unavailable. This code assumes cuda everywhere.")
            exit(-1)
        model = torch.jit.load(model_filepath).cuda().eval()
        input_resolution = model.input_resolution.item()
        context_length = model.context_length.item()
        vocab_size = model.vocab_size.item()
        print("Loaded CLIP Model")
        print("Model parameters:", f"{np.sum([int(np.prod(p.shape)) for p in model.parameters()]):,}")
        print("Input resolution:", input_resolution)
        print("Context length:", context_length)
        print("Vocab size:", vocab_size)
        return model

    return None     
if __name__ == '__main__':
    model = get_model(CLIP)