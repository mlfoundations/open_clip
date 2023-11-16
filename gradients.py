#test file delete 
import torch
from PIL import Image
import open_clip
import torch.nn as nn
import torchvision.transforms as T
import torch.nn.functional as F
import cv2
import matplotlib.pyplot as plt
import numpy as np
import open_clip
import torch
import torch.nn as nn
from PIL import Image
from src.gradcam.hook import Hook
from src.gradcam.utils import get_cnn_modules,show_attention_map


image_name = "test.jpg"


model, _, preprocess = open_clip.create_model_and_transforms('convnext_base', pretrained='laion400m_s13b_b51k')
tokenizer = open_clip.get_tokenizer('convnext_base')

image = preprocess(Image.open("test.jpg")).unsqueeze(0)
transform = T.ToPILImage()

text = tokenizer(["a cat", "a dog"])

cnn_modules= get_cnn_modules(model.visual)
#print(model)
last_layer = cnn_modules[1]
#print(last_layer


#if image.grad is not None:
#    image.grad.data.zero_()

requires_grad = {}
for name, param in model.visual.named_parameters():
    requires_grad[name] = param.requires_grad
    param.requires_grad_(False)

for cnn_layer in cnn_modules:


    with torch.cuda.amp.autocast(), torch.autograd.set_detect_anomaly(True), torch.set_grad_enabled(True), Hook(cnn_layer) as hook:

        image_features = model.encode_image(image)
        text_features = model.encode_text(text)

        normalized_image_features = image_features / torch.linalg.norm(image_features,dim=-1, keepdim=True)

        normalized_text_features = text_features / torch.linalg.norm(text_features,dim=-1, keepdim=True)

        text_probs = (100.0 * normalized_image_features @ normalized_text_features.T)


        #output = model.visual(image)

        #output.backward(model.encode_text(text))

        #text_probs.backward(torch.tensor([[1,0]]))
        text_probs[:, 1].backward()

        grad = hook.gradient.float()

        global_avg_pooled = grad.mean(dim=(2,3),keepdim=True)

        act = hook.activation.float()

        wheigted_act_sum = torch.sum(act*global_avg_pooled,dim=1,keepdim=True)

        #transform(wheigted_act_sum[0]).show()
        grad_cam = torch.clamp(wheigted_act_sum,min=0)
        #transform(grad_cam[0]).show()

        print("max:",torch.max(grad_cam))    
        grad_cam /= torch.max(grad_cam)

        #transform(grad_cam[0]).show()
        grad_cam = grad_cam.squeeze().detach().cpu().numpy()
        show_attention_map(grad_cam,layer_name=cnn_layer,write_to_disk=True)
    

#print("Label probs:", text_probs)  # prints: [[1., 0., 0.]]

#print(model.visual.trunk.stages[-1])
#print(len(cnn_modules))