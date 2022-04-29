import torch 
import clip.clip as clip 
from src.CONST import NUM_LABELS
import torch.nn as nn 

from src.config import CLIP_IMG_EMB_SIZE, CLIP_TEXT_CONTEXT_SIZE, CLIP_IMG_EMB_SIZE, CLIP_TEXT_EMB_SIZE

class CLIP_only_dataset(torch.utils.data.Dataset):

    def __init__(self, data) -> None:
        _, clip_image_preprocess = clip.load('ViT-B/32', jit=False)
        
        self.texts = [] 
        self.images = []
        self.labels = []
        for data_pt in data:
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
        return (self.texts[idx],  self.images[idx]), self.labels[idx]
    

class CLIP_only_Input_transformer:

    def __init__(self) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def transform(self,x,y):
        text_rep = torch.as_tensor(x[0]).long().to(self.device)
        img_rep = x[1].to(self.device)
        y = torch.as_tensor(y).float().to(self.device)
        return (text_rep, img_rep), y

class CLIP_only_Model(nn.Module):

    def __init__(self) -> None:
        super().__init__()
        
        self.clip, _ = clip.load('ViT-B/32', jit=False)
        self.clip = self.clip.cuda()
        for param in self.clip.parameters():
            param.requires_grad = False 
        self.l1 = nn.Sequential(
            nn.Linear(CLIP_IMG_EMB_SIZE + CLIP_TEXT_EMB_SIZE,NUM_LABELS),
            nn.ReLU()
        )
        self.sig = nn.Sigmoid()

    def l2_norm(self, r_factor):
        weights = torch.cat([x.view(-1) for x in self.parameters() if x.requires_grad])
        return r_factor*torch.norm(weights, 2)

    def forward(self, text_rep, img_rep):
        text_emb = self.clip.encode_text(text_rep) #512
        image_emb = self.clip.encode_image(img_rep)  #512 
        x = torch.cat( [text_emb, image_emb] , dim=1).float()
        x = self.l1(x)
        x = self.sig(x)
        pred = torch.round(x)
        return x, pred