import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch.optim as optim

class ContrastiveDataset(Dataset):
    def __init__(self, text_vecs, img_vecs):
        self.text = torch.from_numpy(text_vecs).float()
        self.img = torch.from_numpy(np.array(img_vecs)).float()
        assert len(self.text) == len(self.img), "텍스트/이미지 개수 불일치!"
   
    def __len__(self):
        return len(self.text)
    def __getitem__(self, idx):
        return self.text[idx], self.img[idx]
    

class ContrastiveModel(nn.Module):
    def __init__(self, dim, temperature=0.15):
        super().__init__()
        self.text_proj = nn.Linear(dim, dim, bias=False)
        self.img_proj = nn.Linear(dim, dim, bias=False)
        self.temp = temperature
    
    def forward(self, text_vecs, img_vecs):
        t = F.normalize(self.text_proj(text_vecs), dim=1)
        v = F.normalize(self.img_proj(img_vecs), dim=1)
        logits = (t @ v.T) / self.temp
        return logits


def train(dataset, epoch_size=1, batch_size=1024):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = ContrastiveModel(dim=512).to(device)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    opt = optim.Adam(model.parameters(), lr=1e-3)
    for epoch in range (1, epoch_size + 1):
        total_loss = 0.0
        for text_batch, img_batch in loader:
            text_batch = text_batch.to(device)
            img_batch = img_batch.to(device)
            logits = model(text_batch, img_batch)
            labels = torch.arange(logits.size(0), device=device)
            loss = F.cross_entropy(logits, labels)
            # backpropagation
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item()
        print(f"Epoch {epoch} -> avg loss: {total_loss/len(loader):.4f}")
    return model


class ProjectionEncoder(object):
    def __init__(self, text_encoder, model):
        self.encoder = text_encoder
        self.model = model
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model.eval()
    
    def proj_vec(self, text_vecs, img_vecs):
        text_tensor = torch.from_numpy(text_vecs).float().to(self.device)
        img_tensor = torch.stack(img_vecs).to(self.device)
        with torch.no_grad():
            text_proj = self.model.text_proj(text_tensor)
            text_proj = F.normalize(text_proj, dim=1)
            img_proj = self.model.img_proj(img_tensor)
            img_proj = F.normalize(img_proj, dim=1)
        return text_proj.cpu().numpy(), img_proj.cpu().numpy()

    def encode_query(self, query_text):
        query = self.encoder.encode_query(query_text)
        query = torch.from_numpy(query).float().to(self.device)
        with torch.no_grad():
            query = self.model.text_proj(query)
            query = F.normalize(query, dim=1)
        return query.cpu().numpy()