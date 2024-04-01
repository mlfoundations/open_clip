import torch
from PIL import Image
import open_clip
from training import main, parse_args
import argparse
import yaml

def convert_yaml_to_argv(yaml_dict):
    """
    Convert a dictionary to a list of arguments to be passed to the main function.
    This prevents the need to modify the main function to accept a dictionary as input.
    """
    argv = []
    for key, value in yaml_dict.items():
        argv.append(f'--{key}')
        argv.append(f'{value}')
    return argv

def finetune():
    # Load the configuration file
    with open('hyperparam.yaml') as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    args = convert_yaml_to_argv(config)
    
    main(args)
    # @TODO rewrite the main function to another project and import open_clip
    # and start the finetuning process with COCO2017 -> SVAL Project
    # @TODO add the path to the COCO2017 dataset
    # @TODO add the simplex projection to the model


if __name__ == '__main__':
    finetune()



