import open_clip


def main():
    #'ViT-B-32', 'openai' 
    architecture = 'ViT-B/32' # ResNet50, ViT-B/32
    pretrained = 'openai' 
    model, _, preprocess = open_clip.create_model_and_transforms(architecture, pretrained=pretrained)
    print(model)

    model.to('cuda')
    model.eval()
    context_length = model.context_length
    vocab_size = model.vocab_size

    # Finetuning the model on COCO2017 dataset in the equilibrium/datasets/COCO2017 folder



if __name__ == "__main__":
    main()