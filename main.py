import argparse
from framework.utils import *
from exp.ContrastiveLearning import *


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='clip_train.yaml', help='config file')
    args = parser.parse_args()

    # --------------------
    # Module Instantiation
    # --------------------
    config = read_config(args.config)
    defaults = config["default"]
    
    # image encoder
    img_enc = instantiate_module('img_enc', config["img_enc"], defaults)
    # text encoder
    text_enc = instantiate_module('text_enc', config["text_enc"], defaults)
    # dataset
    dataset = instantiate_module('dataset', config["dataset"], defaults)
    # preprocessor
    preprocessor_list = [instantiate_module('preprocessor', preprocessor_config, defaults) for preprocessor_config in config["preprocessor"]]
    
    
    if defaults["exp"] == "ContrastiveLearning":
        ContrastiveLearning(img_enc, text_enc, dataset, preprocessor_list, config)

if __name__ == "__main__":
    main()
