## Text-Based Image Retrieval

**Task**  
: Retrieve a relevant set of photos when given a sentence query.

**Dataset**  
: [MS COCO dataset](https://cocodataset.org/#download) - 2017 Train/Val images


## 1. Baseline Model

Text-Encoder: **TF-IDF**  
Image-Encoder: **pretrained-ResNet18**  
Loss function: InfoNCE loss


## 2. Transformer-based Approach

Text-Encoder: **BERT**  
Image-Encoder: **ViT**  
Loss function: InfoNCE loss


## 3. Mixed-Up Models

### (1) mixed-up model 1
Text-Encoder: **OpenCLIP-BERT**  
Image-Encoder: **pretrained_ResNet18**

### (2) mixed-up model 2
Text-Encoder: **TF-IDF**  
Image-Encoder: **pretrained_ResNet18**

## 4. Recall@K & Time-complexity (encoding) comparing Master Table

*Note: text & image encoding time is for validation dataset (5000 dataset).*

|Model|Recall@1|Recall@5|Recall@10|Text-Encoding Time|Image-Encoding Time|
|:-------:|---:|---:|---:|---:|---:|
|baseline (TF-IDF + ResNet18)|0.114|0.317|0.433|~0 sec| 46 sec|
|Pretrained BERT + ViT|0.000|0.001|0.002|34 sec|1 min 10 sec|
|Pretrained BERT + ViT (train proj heads)|0.010|0.042|0.075|34 sec|1 min 10 sec|
|Pretrained BERT + ViT (full fine-tune)|0.017|0.069|0.122|34 sec|1 min 9sec|
|Pretrained BERT + OpenCLIP ViT (train proj heads)|0.000|0.001|0.002|34 sec|49 sec|
|OpenCLIP BERT + Pretrained ViT (train proj heads)|0.000|0.000|0.001|41 sec|1 min 10 sec|
|OpenCLIP BERT + OpenCLIP ViT|0.299|0.540|0.651|41 sec|50 sec|
|mixed-up 1 (OpenCLIP BERT + ResNet18)|0.219|0.487|0.611|53 sec|44 sec|
|mixed-up 2 (TF-IDF + OpenCLIP ViT)|0.000|0.001|0.002|~0 sec|50 sec|

![Recall@K Comparison Graph](https://github.com/SongWon03/20251R0136COSE47400/blob/main/Comparison%20of%20Recall@K%20for%20all%20models.png?raw=true)
