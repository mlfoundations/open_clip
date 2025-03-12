from loss.mse import *

def ContrastiveLearning(img_enc, text_enc, dataset, preprocessor_list, config):
    
    # you can load your training config here
    # {'epochs': 10, 'batch_size': 32, ...}
    train_config = config["train"]
    
    print("Contrastive Learning")