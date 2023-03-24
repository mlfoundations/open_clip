import argparse

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch import nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

import open_clip
from open_clip.factory import get_tokenizer
from training.scheduler import cosine_lr
from training.train import AverageMeter
import evaluate
import random
import json


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
with open("kilogram/development/texts/controlled/whole+black.json") as f:
  file_contents = json.load(f)
print(file_contents.keys())

def sample(texts, images, r, k = 10):
  sampled_images = [images[r]]
  sampled_text = [texts[r]]
  
  while(len(sampled_images) < k):
    r = random.randint(0, len(images)-1)
    if(images[r] in sampled_images or texts[r] in sampled_text):
      continue
    sampled_images.append(images[r])
    sampled_text.append(texts[r])
  return (sampled_text, sampled_images)

class CLIPMultimodalClassifier(nn.Module):
    def __init__(self, encoder):
        super().__init__()

        self.encoder = encoder
    def forward(self, image, text):
        # CLIP doesn't have a multimodal encoder, so we concatenate the features
        text_features = self.encoder.encode_text(text)
        image_features = self.encoder.encode_image(image)
        
        return torch.dot(text_features, image_features)
model, preprocess_train, preprocess_val = open_clip.factory.create_model_and_transforms(
    "ViT-B-32-quickgelu",
    'laion400m_e32',
    precision="amp",
    device=device,
)
clf_cls = CLIPMultimodalClassifier
clf = clf_cls(model).to(device)
transforms = preprocess_val
tokenizer = get_tokenizer("ViT-B-32-quickgelu")

total_correct = 0
total_games = 0
for i in range(len(file_contents['texts'])):
  sample_first = sample(file_contents['texts'], file_contents['images'], i)
  images = []
  texts = []

  for i in range(len(sample_first[0])):
    image = Image.open("kilogram/development/images/black/" + sample_first[1][i] + ".png")

    images.append(transforms(image))
    texts.append(sample_first[0][i])
  image_input = torch.tensor(np.stack(images)).to(device)
  text_tokens = tokenizer(texts).to(device)
  with torch.no_grad():
    image_features = clf.encoder.encode_image(image_input).float()
    text_features = clf.encoder.encode_text(text_tokens).float()
  similarity = np.dot(text_features.cpu().numpy(), image_features.cpu().numpy().T)

  k = 10
  total_games += k
  for i in range(k):
    if(np.argmax(similarity[i]) == i):
      total_correct += 1
  
  print(total_correct/total_games)
