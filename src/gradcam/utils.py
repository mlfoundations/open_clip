import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
from torch import nn

def load_image(img_path, resize=None):
    image = Image.open(img_path).convert("RGB")
    if resize is not None:
        image = image.resize((resize, resize))
    return np.asarray(image).astype(np.float32) / 255.0


def show_attention_map(heatmap, image_name:str, layer_name:str="",write_to_disk:bool=False):
    _, axes = plt.subplots(1, 3, figsize=(15, 8))
    axes[0].matshow(heatmap.squeeze())
    img = cv2.imread(image_name)
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    result_img = (heatmap * 0.4 + img).astype(np.float32)
    axes[1].imshow(img[..., ::-1])
    axes[2].imshow((result_img / 255)[..., ::-1])
    for ax in axes:
        ax.axis("off")
    plt.show()
    if write_to_disk:
        plt.savefig('/imgs/'+layer_name+".jpg")

        
def get_cnn_modules(module,cnn_module_list=[]):
    for child in module.children():
        if type(child) is nn.Conv2d:
            cnn_module_list.append(child)
        elif child.children() is not None:
            cnn_module_list = get_cnn_modules(child,cnn_module_list)
    return cnn_module_list