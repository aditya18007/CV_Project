import torch 
import clip.clip as clip 
from src.CONST import NUM_LABELS
import torch.nn as nn 
from transformers import DistilBertModel
from transformers import DistilBertTokenizer 
from src.config import DISTIL_BERT_MODEL, BERT_MAX_TOKENS, BERT_EMBEDDING_SIZE
from src.config import CLIP_IMG_EMB_SIZE, CLIP_TEXT_CONTEXT_SIZE, CLIP_IMG_EMB_SIZE, CLIP_TEXT_EMB_SIZE
import tqdm 

class face_Model(torch.utils.data.Dataset):
    tokenizer_params = { 'padding':'max_length', 
                         'max_length':BERT_MAX_TOKENS, 
                         'truncation':True,
                         'return_tensors':"pt"
                        }
    def __init__(self, data, clip_image_preprocess):
        
        tokenizer = DistilBertTokenizer.from_pretrained(DISTIL_BERT_MODEL)
        self.tokenized = [tokenizer( data_pt['text'], **self.tokenizer_params) for data_pt in tqdm.tqdm(data, desc="distilBERT Tokenizer")]
        self.labels = [data_pt['encoded_labels'] for data_pt in data]
        
        self.texts = [] 
        self.images = []
        for data_pt in tqdm.tqdm(data, desc="CLIP preprocessing"):
            text_clip_rep = clip.tokenize(data_pt['text'], context_length=CLIP_TEXT_CONTEXT_SIZE, truncate=True).squeeze()
            self.texts.append(text_clip_rep)

            image_clip_rep = clip_image_preprocess(data_pt['image_PIL'])
            self.images.append(image_clip_rep)

            self.labels.append(data_pt['encoded_labels'])

        assert(len(self.texts) == len(self.images))
        self.length = len(self.texts)
    
    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        return (self.tokenized[idx], self.texts[idx],  self.images[idx]), self.labels[idx]
    