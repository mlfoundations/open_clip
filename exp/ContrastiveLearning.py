
def ContrastiveLearning(img_enc, text_enc, dataset, preprocessor_list, config):
    train_config = config["train"]
    
    print("Contrastive Learning")