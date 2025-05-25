# 20251R0136COSE47400

## Text-Based Image Retrieval

**Task**  
: Retrieve a relevant set of photos when given a sentence query.

**Dataset**  
: [MS COCO dataset](https://cocodataset.org/#download) - 2017 Train/Val images


## 1. Baseline Model

Text-Encoder: **TF-IDF**  
Image-Encoder: **pretrained-ResNet18**  
Loss function: InfoNCE loss (with small batch)  
Evaluation Metric:  


## 2. Improved Model

Text-Encoder: **BERT**  
Image-Encoder: **ViT**  
Loss function: InfoNCE loss  
Evaluation Metric:  


## 3. Mixed-Up Models


## 4. Accuracy & Time-Complexity comparing Table

|Model|Accuracy|Text-Encoding Time|Image-Encoding Time|Computing Source|
|---:|---:|---:|---:|---:|
|TF-IDF & ResNet18 (**baseline**)||8 sec (~110,000 captions)|27 min (~110,000 images)|T4 GPU|
|BERT & ViT (**improved**)|||||
||||||
