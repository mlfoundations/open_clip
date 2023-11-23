import cv2
import matplotlib.pyplot as plt

import numpy as np
import os
from PIL import Image
from torch import nn

def load_image(img_path, resize=None):
    image = Image.open(img_path).convert("RGB")
    if resize is not None:
        image = image.resize((resize, resize))
    return np.asarray(image).astype(np.float32) / 255.0

def check_and_create_path(path):
    # Check if the path exists
    if not os.path.exists(path):
        # If the path doesn't exist, create it
        os.makedirs(path)
        print(f"Path '{path}' created.")

def show_attention_map(heatmap, image_name:str, layer_name:str="",caption:str="",write_to_disk:bool=False,idx:str=False):
    _, axes = plt.subplots(1, 1, figsize=(15, 8))
    axes.matshow(heatmap.squeeze())
    '''
    img = cv2.imread(image_name)
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    result_img = (heatmap * 0.4 + img).astype(np.float32)
    axes[1].imshow(img[..., ::-1])
    axes[2].imshow((result_img / 255)[..., ::-1])
    '''
    #for ax in axes:
    #    ax.axis("off")
    if write_to_disk:
        cwd =  os.getcwd()
        dict_path = cwd + '/imgs/heatmaps/'+caption+"/"
        check_and_create_path(dict_path)
        img_name = idx+layer_name+".jpg"
        path = dict_path+img_name
        plt.savefig(path)

    else:
        plt.show()

        
def get_cnn_modules(module,cnn_module_list=[]):
    for child in module.children():
        if type(child) is nn.Conv2d:
            cnn_module_list.append(child)
        elif child.children() is not None:
            cnn_module_list = get_cnn_modules(child,cnn_module_list)
    return cnn_module_list

def get_all_layers(module,layer_list=[]):
    for child in module.children():
        if len(list(child.children()))==0:
            layer_list.append(child)
        else:
            layer_list = get_all_layers(child,layer_list)
    return layer_list