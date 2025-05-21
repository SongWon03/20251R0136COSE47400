from pycocotools.coco import COCO
from sklearn.feature_extraction.text import TfidfVectorizer
import torch
import torchvision.models as models
import torchvision.transforms as transforms
import os
from PIL import Image
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import numpy as np
from IPython.display import display


class TF_IDF(object):
    def __init__(self, data_type, max_features, use_embedding=False):
        self.data_type = data_type
        self.data_dir = '/content/coco/'
        self.ann_file = '{}/annotations/annotations/captions_{}.json'.format(self.data_dir, data_type)
        self.coco_caps = COCO(self.ann_file)
        self.img_ids = self.coco_caps.getImgIds()
        self.max_features = max_features
        self.use_embedding = use_embedding
        self.vectorizer = TfidfVectorizer(
            max_features= self.max_features,
            ngram_range=(1, 1),   # use uni-gram
            lowercase=True,
            stop_words='english')
        self.svd = None
      
    def fit_tfidf(self):
        captions = []
        ids_for_matrix = []
        for img_id in self.img_ids:
            ann_ids = self.coco_caps.getAnnIds(imgIds=[img_id])
            anns = self.coco_caps.loadAnns(ann_ids)
            text = ' '.join([ann['caption'] for ann in anns])
            captions.append(text)
            ids_for_matrix.append(img_id)
        # apply TF-IDF encoding
        tfidf_matrix = self.vectorizer.fit_transform(captions)
        if self.use_embedding:
            self.svd = TruncatedSVD(n_components=512, random_state=42)
            tfidf_matrix = self.svd.fit_transform(tfidf_matrix)
        return tfidf_matrix, ids_for_matrix

    def encode_query(self, query_text):
        # apply TF-IDF encoding to query
        q_tfidf = self.vectorizer.transform([query_text])
        if self.use_embedding:
            if self.svd is None:
                raise RuntimeError('Call fit_tfidf(use_embedding=True) before encoding query.')
            return self.svd.transform(q_tfidf)
        else:
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
    
    
def cos_similarity(data_type, img_features, ids_for_matrix, query_vector):
    img_feats_np = np.stack([f.numpy() for f in img_features])
    similarities = cosine_similarity(query_vector, img_feats_np)[0]
    # select top 5 results
    top5_idx = similarities.argsort()[-5:][::-1]
    top5_img_ids = [ids_for_matrix[i] for i in top5_idx]
    top5_scores = similarities[top5_idx]
    # visualize
    data_dir = '/content/coco'
    img_dir = '{}/{}/{}'.format(data_dir, data_type, data_type)
    for img_id, score in zip(top5_img_ids, top5_scores):
        filename = f"{img_id:012d}.jpg"
        path = os.path.join(img_dir, filename)
        img = Image.open(path).convert('RGB')
        print(f"ID:{img_id} | Cosine sim: {score:.3f}")
        display(img)