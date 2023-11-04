#delete

import torch
from PIL import Image
import open_clip
import torch.nn as nn
model, _, preprocess = open_clip.create_model_and_transforms('convnext_base', pretrained='laion400m_s13b_b51k')
tokenizer = open_clip.get_tokenizer('convnext_base')

image = preprocess(Image.open("test.jpg")).unsqueeze(0)
text = tokenizer(["a dog", "a cat",])


with torch.cuda.amp.autocast(), torch.autograd.set_detect_anomaly(True), torch.set_grad_enabled(True):
    image_features = model.encode_image(image)
    text_features = model.encode_text(text)
    
    text_features.retain_grad()
    image_features.retain_grad()

    normalized_image_features = image_features / torch.linalg.norm(image_features,dim=-1, keepdim=True)
    normalized_image_features.retain_grad()

    normalized_text_features = text_features / torch.linalg.norm(text_features,dim=-1, keepdim=True)
    normalized_text_features.retain_grad()

    text_probs = (100.0 * normalized_image_features @ normalized_text_features.T).softmax(dim=-1)
    text_probs.retain_grad()
    labels = torch.tensor([[0., 1.]])

    # Compute a loss based on text_probs and labels
    criterion = nn.CrossEntropyLoss()
    loss = criterion(text_probs, labels)
    
    loss.retain_grad()
    # Perform the backward pass to compute gradients
    loss.backward()

    #if we dont want, scalar gradient
    #gradient = torch.ones_like(image_features)  # Create a gradient tensor of the same shape
    #image_features.retain_grad()
    #print(image_features.backward(gradient))
    

print(loss.grad.size())
print(text_probs.grad.size())
print(normalized_image_features.grad.size())
print(image_features.grad.size())
for name, param in model.named_parameters():
    print(name, param.size())
for param in model.parameters():
    if param.grad is not None:
        print(param.grad.size())
    else:
        print(type(param))
print("Label probs:", text_probs)  # prints: [[1., 0., 0.]]