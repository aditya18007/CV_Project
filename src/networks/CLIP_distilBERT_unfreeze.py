import torch 
import clip.clip as clip 
from src.CONST import NUM_LABELS
import torch.nn as nn 
from transformers import DistilBertModel
from transformers import DistilBertTokenizer 
from src.config import DISTIL_BERT_MODEL, BERT_MAX_TOKENS, BERT_EMBEDDING_SIZE
from src.config import CLIP_IMG_EMB_SIZE, CLIP_TEXT_CONTEXT_SIZE, CLIP_IMG_EMB_SIZE, CLIP_TEXT_EMB_SIZE
import tqdm 

class CLIP_dBERT_unfreezed_layers_dataset(torch.utils.data.Dataset):
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
    
class CLIP_dBERT_unfreezed_layers_Input_transformer:

    def __init__(self) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def transform(self,x,y):
        #(bert tokenized, clip_txt, clip_img)
        id = x[0]['input_ids'].squeeze(1).to(self.device)
        mask = x[0]['attention_mask'].to(self.device)
        text_rep_clip = x[1].to(self.device)
        img_rep_clip = x[2].to(self.device)
        y = torch.as_tensor(y).float().to(self.device)
        return (id, mask, text_rep_clip, img_rep_clip), y
    

class CLIP_dBERT_unfreezed_layers_Model(nn.Module):

    def __init__(self, clip_model) -> None:
        super().__init__()
        self.distil_BERT = DistilBertModel.from_pretrained(DISTIL_BERT_MODEL)
        for param in self.distil_BERT.parameters():
            param.requires_grad = False 
        self.clip = clip_model.cuda()
        for param in self.clip.parameters():
            param.requires_grad = False 
        
        for name, param in self.distil_BERT.named_parameters():
            if ('transformer.layer.5' in name):
                param.requires_grad = True
                print(f"Unfreezing - {name}")

        self.l1 = nn.Sequential(
            nn.Linear(CLIP_IMG_EMB_SIZE + CLIP_TEXT_EMB_SIZE + BERT_EMBEDDING_SIZE,128),
            nn.Dropout(0.3),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128,NUM_LABELS)
        )
        self.sig = nn.Sigmoid()

    def l2_norm(self, r_factor):
        weights = torch.cat([x.view(-1) for x in self.parameters() if x.requires_grad])
        return r_factor*torch.norm(weights, 2)

    def forward(self, id, mask, text_rep_clip, img_rep_clip):
        pooled_output = self.distil_BERT(input_ids=id, attention_mask=mask)
        dBERT_text = pooled_output.last_hidden_state #768
        dBERT_text = dBERT_text[:, 0, :]
        text_emb_clip = self.clip.encode_text(text_rep_clip) #512
        image_emb_clip = self.clip.encode_image(img_rep_clip)  #512 
        x = torch.cat( [dBERT_text, text_emb_clip, image_emb_clip], dim=1)
        x = self.l1(x)
        x = self.sig(x)
        pred = torch.round(x)
        return x, pred 