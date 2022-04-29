import torch 
import clip.clip as clip 
from src.CONST import NUM_LABELS
import torch.nn as nn 
from transformers import DistilBertModel
from transformers import DistilBertTokenizer 
from src.config import DISTIL_BERT_MODEL, BERT_MAX_TOKENS, BERT_EMBEDDING_SIZE
from src.config import CLIP_IMG_EMB_SIZE, CLIP_TEXT_CONTEXT_SIZE, CLIP_IMG_EMB_SIZE, CLIP_TEXT_EMB_SIZE
import tqdm 
import cv2 
from PIL import Image
from facenet_pytorch.models.utils.detect_face import extract_face
from torchvision import transforms

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


class face_Model(torch.utils.data.Dataset):
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
                continue
            boxes = self.get_boxes(image)
            if boxes is None:
                continue
            
            emb = self.get_embeddings(image, boxes)
            if emb is None:
                continue 
                    
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
        embeddings =  self.resnet(facenet_preprocess(faces.cuda()))
        embeddings = torch.mean(embeddings, dim=0)
        return embeddings
        
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
        return (self.tokenized[idx], self.texts[idx],  self.images[idx]), self.labels[idx]
    