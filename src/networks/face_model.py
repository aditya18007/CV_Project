import torch 
import clip.clip as clip 
from src.CONST import NUM_LABELS
import torch.nn as nn 
from transformers import DistilBertModel
from transformers import DistilBertTokenizer 
from src.config import DISTIL_BERT_MODEL, BERT_MAX_TOKENS, BERT_EMBEDDING_SIZE
from src.config import CLIP_IMG_EMB_SIZE, CLIP_TEXT_CONTEXT_SIZE, CLIP_IMG_EMB_SIZE, CLIP_TEXT_EMB_SIZE, FACE_EMBEDDING_SIZE
import tqdm 
import cv2 
from PIL import Image
from facenet_pytorch.models.utils.detect_face import extract_face
from torchvision import transforms
import sys

class Whitening(object):
    """
    Whitens the image.
    Source:https://github.com/arsfutura/face-recognition/blob/master/face_recognition/face_features_extractor.py
    """

    def __call__(self, img):
        mean = img.mean()
        std = img.std()
        std_adj = std.clamp(min=1.0 / (float(img.numel()) ** 0.5))
        y = (img - mean) / std_adj
        return y


class face_model_dataset(torch.utils.data.Dataset):
    tokenizer_params = { 'padding':'max_length', 
                         'max_length':BERT_MAX_TOKENS, 
                         'truncation':True,
                         'return_tensors':"pt"
                        }
    def __init__(self, data, clip_image_preprocess, mtcnn, resnet):

        self.mtcnn = mtcnn 
        self.resnet = resnet 
        tokenizer = DistilBertTokenizer.from_pretrained(DISTIL_BERT_MODEL)
        self.tokenized = [tokenizer( data_pt['text'], **self.tokenizer_params) for data_pt in tqdm.tqdm(data, desc="distilBERT Tokenizer")]
        self.labels = [data_pt['encoded_labels'] for data_pt in data]
        
        self.texts = [] 
        self.images = []
        self.face_embeddings = []

        for data_pt in tqdm.tqdm(data, desc="CLIP preprocessing"):
            text_clip_rep = clip.tokenize(data_pt['text'], context_length=CLIP_TEXT_CONTEXT_SIZE, truncate=True).squeeze()
            self.texts.append(text_clip_rep)

            image_clip_rep = clip_image_preprocess(data_pt['image_PIL'])
            self.images.append(image_clip_rep)

            self.labels.append(data_pt['encoded_labels'])
            
            image = self.read_image(data_pt['image_path'])
            
            if image is None:
                print(f"No image at {data_pt['image_path']}")
                sys.exit(-1)

            boxes = self.get_boxes(image)
            emb_fallback = torch.zeros( (512) )
            if boxes is None:
                self.face_embeddings.append(emb_fallback)
                continue 

            emb = self.get_embeddings(image, boxes)
            if emb is None:
                self.face_embeddings.append(emb_fallback)
                continue
            self.face_embeddings.append(emb)

        assert(len(self.texts) == len(self.images))
        self.length = len(self.texts)
    
    def read_image(self,path):
        try:
            image = Image.fromarray(cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB))
        except cv2.error:
            return None
        return image 

    def get_embeddings(self, image, boxes):
        extracted_faces = [extract_face(image, bb) for bb in boxes]
        faces = torch.stack(extracted_faces)
        facenet_preprocess = transforms.Compose([Whitening()])
        embeddings =  self.resnet(facenet_preprocess(faces.cuda())).cpu().detach()
        embeddings = torch.mean(embeddings, dim=0)
        return embeddings.squeeze()
        
    def get_boxes(self, image):
        boxes, prob = self.mtcnn.detect(image)
        if boxes is None :
            return None
        
        num_boxes = len(prob)
        final_boxes = []
        
        for i in range(num_boxes):
            if prob[i] > 0.8:
                final_boxes.append(boxes[i])
        
        if len(final_boxes) == 0:
            return None 

        return final_boxes


    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        return (self.tokenized[idx], self.texts[idx],  self.images[idx], self.face_embeddings[idx]), self.labels[idx]

class face_model_input_transformer():

    def __init__(self) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def transform(self,x,y):
        #(bert tokenized, clip_txt, clip_img)
        id = x[0]['input_ids'].squeeze(1).to(self.device)
        mask = x[0]['attention_mask'].to(self.device)
        text_rep_clip = x[1].to(self.device)
        img_rep_clip = x[2].to(self.device)
        face_emb = x[3].to(self.device)
        y = torch.as_tensor(y).float().to(self.device)
        return (id, mask, text_rep_clip, img_rep_clip, face_emb), y

class face_model(nn.Module):

    def __init__(self, clip_model) -> None:
        super().__init__()
        self.distil_BERT = DistilBertModel.from_pretrained(DISTIL_BERT_MODEL)
        for param in self.distil_BERT.parameters():
            param.requires_grad = False 
        self.clip = clip_model.cuda()
        for param in self.clip.parameters():
            param.requires_grad = False 
        self.l1 = nn.Sequential(
            nn.Linear(FACE_EMBEDDING_SIZE+CLIP_IMG_EMB_SIZE + CLIP_TEXT_EMB_SIZE + BERT_EMBEDDING_SIZE,128),
            nn.Dropout(0.3),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128,NUM_LABELS)
        )
        self.sig = nn.Sigmoid()

    def l2_norm(self, r_factor):
        weights = torch.cat([x.view(-1) for x in self.parameters() if x.requires_grad])
        return r_factor*torch.norm(weights, 2)

    def forward(self, id, mask, text_rep_clip, img_rep_clip, face_emb):
        pooled_output = self.distil_BERT(input_ids=id, attention_mask=mask)
        dBERT_text = pooled_output.last_hidden_state #768
        dBERT_text = dBERT_text[:, 0, :]
        text_emb_clip = self.clip.encode_text(text_rep_clip) #512
        image_emb_clip = self.clip.encode_image(img_rep_clip)  #512 
        x = torch.cat( [dBERT_text, text_emb_clip, image_emb_clip, face_emb], dim=1)
        x = self.l1(x)
        x = self.sig(x)
        pred = torch.round(x)
        return x, pred 