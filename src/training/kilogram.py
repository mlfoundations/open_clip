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
import os
import json

from torch.utils.data import Dataset, DataLoader, Sampler

#Download data:
#git lfs install
#git clone https://huggingface.co/datasets/lil-lab/kilogram

class ValidationDataSet(Dataset):
	def __init__(self, image_path, data, tokenizer, image_processor):
		'''
		Requires:
		[image_path]: path to images folder
		[data]: contains targets, texts, image filenames
		'''
		self.image_path = image_path
		self.images_n = data['images']
		self.texts = data['texts']
		self.targets = data['targets'] 
		self.tokenizer = tokenizer
		self.image_processor = image_processor
	def __len__(self):
		'''__len__ returns the number of samples in the dataset.
		:returns: number of (image, annotation) pairs in dataset
		:rtype: int
		'''
		return len(self.texts) 

	def __getitem__(self, idx):
		'''
		__getitem__ returns the tensor, output pair for a given index
		  :param idx: index within dataset to return
		  :type idx: int
		  :returns: image tensor, text tensor
		  :rtype: tensor, tensor

		'''
		image_file = self.images_n[idx]
		image_path = os.path.join(self.image_path, image_file) + '.png' if not image_file.endswith('.png') else os.path.join(self.image_path, image_file)
		texts = self.texts[idx]
		text_tokens = torch.squeeze(self.tokenizer(texts))
		images = self.image_processor(Image.open(image_path))
		target = self.targets[idx]
		return text_tokens, images

class CLIPMultimodalClassifier(nn.Module):
	def __init__(self, encoder):
		super().__init__()
		self.encoder = encoder
	def forward(self, image, text):
		# CLIP doesn't have a multimodal encoder, so we concatenate the features
		text_features = self.encoder.encode_text(text)
		image_features = self.encoder.encode_image(image)

		return torch.dot(text_features, image_features)

def main(args):     
	device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
	with open("kilogram/development/texts/controlled/whole+black.json") as f:
		file_contents = json.load(f)
	model, preprocess_train, preprocess_val = open_clip.factory.create_model_and_transforms(
		"ViT-B-32-quickgelu",
		'openai',
		precision="amp",
		device=device,
	)
	clf_cls = CLIPMultimodalClassifier
	clf = clf_cls(model).to(device)

	transforms = preprocess_val
	tokenizer = get_tokenizer("ViT-B-32-quickgelu")

	dsval = ValidationDataSet("kilogram/development/images/black/", file_contents, tokenizer, transforms)
	dlval = DataLoader(dsval, batch_size=10, shuffle=False, drop_last=True)

	total_games = 0
	total_correct = 0
	for text_tokens, image_input in dlval:
		text_tokens = torch.tensor(text_tokens).to(device)
		image_input = torch.tensor(image_input).to(device)
		clf.eval()
		with torch.no_grad():
			image_features = clf.encoder.encode_image(image_input).float()
			text_features = clf.encoder.encode_text(text_tokens).float()
		similarity = np.dot(text_features.cpu().numpy(), image_features.cpu().numpy().T)
		total_games += 2
		if(np.argmax(similarity, axis = 0)[0] == 0):
			total_correct += 1
		if(np.argmax(similarity, axis = 1)[0] == 0):
			total_correct += 1
	print(total_correct/total_games)

if __name__ == "__main__":
	main(sys.argv[1:])
