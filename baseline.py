from pycocotools.coco import COCO
from sklearn.feature_extraction.text import TfidfVectorizer
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import os
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from IPython.display import display


class TF_IDF(nn.Module):
    def __init__(self, data_type, max_features, embed_dim=512, use_fc=False, device=None):
        super().__init__()
        self.data_type = data_type
        self.data_dir = '/content/coco/'
        self.ann_file = '{}/annotations/annotations/captions_{}.json'.format(self.data_dir, data_type)
        self.coco_caps = COCO(self.ann_file)
        self.img_ids = self.coco_caps.getImgIds()
        self.max_features = max_features
        self.embed_dim = embed_dim
        self.use_fc = use_fc
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.vectorizer = TfidfVectorizer(
            max_features= self.max_features,
            ngram_range=(1, 1),   # use uni-gram
            lowercase=True,
            stop_words='english')
        if self.use_fc:
            self.fc = nn.Linear(self.max_features, self.embed_dim, bias=True).to(self.device)

    def fit_tfidf(self):
        captions = []
        ids_for_matrix = []
        for img_id in self.img_ids:
            ann_ids = self.coco_caps.getAnnIds(imgIds=[img_id])
            anns = self.coco_caps.loadAnns(ann_ids)
            text = ' '.join([ann['caption'] for ann in anns])
            captions.append(text)
            ids_for_matrix.append(img_id)
        # apply TF-IDF encoding & FC layer
        tfidf_matrix = self.vectorizer.fit_transform(captions)
        if self.use_fc:
            tfidf_dense = torch.from_numpy(tfidf_matrix.toarray()).float().to(self.device)
            emb = self.fc(tfidf_dense)
        return emb.detach().cpu().numpy(), ids_for_matrix

    def encode_query(self, query_text):
        # apply TF-IDF encoding to query
        q_tfidf = self.vectorizer.transform([query_text])
        if self.use_fc:
            q_dense = torch.from_numpy(q_tfidf.toarray()).float().to(self.device)
            q_emb = self.fc(q_dense)
            return q_emb.detach().cpu().numpy()
        return q_tfidf


class ImageEncoder(object):
    def __init__(self, data_type, ids_for_matrix, device=None):
        self.data_type = data_type
        self.data_dir = '/content/coco'
        self.img_dir = '{}/{}/{}'.format(self.data_dir, self.data_type, self.data_type)
        self.ids_for_matrix = ids_for_matrix
        self.preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.load_resnet().to(self.device)

    def load_resnet(self):
        resnet = models.resnet18(pretrained=True)
        resnet = torch.nn.Sequential(*list(resnet.children())[:-1])   # remove FC layer for feature extraction
        resnet.eval()
        return resnet

    def fit_resnet(self):
        img_features = []
        for img_id in self.ids_for_matrix:
            filename = f"{img_id:012d}.jpg"
            path = os.path.join(self.img_dir, filename)
            img = Image.open(path).convert('RGB')
            batch = self.preprocess(img).unsqueeze(0)   # (3, 224, 224) -> (1, 3, 224, 224) for pyTorch
            batch = batch.to(self.device)
            with torch.no_grad():
                feature = self.model(batch).squeeze()  # (512,)
            img_features.append(feature.cpu())
        return img_features


def q_cos_similarity(data_type, text_encoder, feature_source, ids_for_matrix, query_text, top_k=5):
    q_vec = text_encoder.encode_query(query_text)
    # feature source to numpy array
    if isinstance(feature_source, np.ndarray):
        features = feature_source
    else:
        features = np.stack([f.cpu().numpy() for f in feature_source])
    similarities = cosine_similarity(q_vec, features)[0]
    # select top 5 results
    top_idx = similarities.argsort()[-top_k:][::-1]
    top_img_ids = [ids_for_matrix[i] for i in top_idx]
    top_scores = similarities[top_idx]
    # visualize
    data_dir = '/content/coco'
    img_dir = '{}/{}/{}'.format(data_dir, data_type, data_type)
    for img_id, score in zip(top_img_ids, top_scores):
        filename = f"{img_id:012d}.jpg"
        path = os.path.join(img_dir, filename)
        img = Image.open(path).convert('RGB')
        print(f"ID:{img_id} | Cosine sim: {score:.3f}")
        display(img)
