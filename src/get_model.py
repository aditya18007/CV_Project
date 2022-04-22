import os 
import wget 

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

    if not os.path.exists(f'Models/{model_name}'):
        wget.download(Links[model_name], out=f'Models/{model_name}.pt')
    
if __name__ == '__main__':
    get_model(CLIP)