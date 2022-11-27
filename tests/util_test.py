
import os
import random
import numpy as np
from PIL import Image
import torch
import open_clip

os.environ['CUDA_VISIBLE_DEVICES'] = ''

def seed_all(seed = 0):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=False)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def inference_text(model, model_name, batches):
    y = []
    tokenizer = open_clip.get_tokenizer(model_name)
    with torch.no_grad():
        for x in batches:
            x = tokenizer(x)
            y.append(model.encode_text(x))
        return torch.stack(y)

def inference_image(model, preprocess_val, batches):
    y = []
    with torch.no_grad():
        for x in batches:
            x = torch.stack([preprocess_val(img) for img in x])
            y.append(model.encode_image(x))
        return torch.stack(y)

def random_image_batch(batch_size, size):
    h, w = size
    data = np.random.randint(255, size = (batch_size, h, w, 3), dtype = np.uint8)
    return [ Image.fromarray(d) for d in data ]

def random_text_batch(batch_size, min_length = 75, max_length = 75):
    t = open_clip.tokenizer.SimpleTokenizer()
    # every token decoded as string, exclude SOT and EOT, replace EOW with space
    token_words = [
            x[1].replace('</w>', ' ')
            for x in t.decoder.items()
            if x[0] not in t.all_special_ids
    ]
    # strings of randomly chosen tokens
    return [
        ''.join(random.choices(
                token_words,
                k = random.randint(min_length, max_length)
        ))
        for _ in range(batch_size)
    ]

def create_random_text_data(
        path,
        min_length = 75,
        max_length = 75,
        batches = 1,
        batch_size = 1
):
    text_batches = [
            random_text_batch(batch_size, min_length, max_length)
            for _ in range(batches)
    ]
    print(f"{path}")
    torch.save(text_batches, path)

def create_random_image_data(path, size, batches = 1, batch_size = 1):
    image_batches = [
            random_image_batch(batch_size, size)
            for _ in range(batches)
    ]
    print(f"{path}")
    torch.save(image_batches, path)

def get_data_dirs(make_dir = True):
    data_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'data')
    input_dir = os.path.join(data_dir, 'input')
    output_dir = os.path.join(data_dir, 'output')
    if make_dir:
        os.makedirs(input_dir, exist_ok = True)
        os.makedirs(output_dir, exist_ok = True)
    assert os.path.isdir(data_dir), f"data directory missing, expected at {input_dir}"
    assert os.path.isdir(data_dir), f"data directory missing, expected at {output_dir}"
    return input_dir, output_dir

def create_test_data_for_model(
        model_name,
        pretrained = None,
        precision = 'fp32',
        jit = False,
        pretrained_hf = False,
        force_quick_gelu = False,
        create_missing_input_data = True,
        batches = 1,
        batch_size = 1,
        overwrite = False
):
    model_id = f'{model_name}_{pretrained or pretrained_hf}_{precision}'
    input_dir, output_dir = get_data_dirs()
    output_file_text = os.path.join(output_dir, f'{model_id}_random_text.pt')
    output_file_image = os.path.join(output_dir, f'{model_id}_random_image.pt')
    text_exists = os.path.exists(output_file_text)
    image_exists = os.path.exists(output_file_image)
    if not overwrite and text_exists and image_exists:
        return
    seed_all()
    model, _, preprocess_val = open_clip.create_model_and_transforms(
            model_name,
            pretrained = pretrained,
            precision = precision,
            jit = jit,
            force_quick_gelu = force_quick_gelu,
            pretrained_hf = pretrained_hf
    )
    # text
    if overwrite or not text_exists:
        input_file_text = os.path.join(input_dir, 'random_text.pt')
        if create_missing_input_data and not os.path.exists(input_file_text):
            create_random_text_data(
                    input_file_text,
                    batches = batches,
                    batch_size = batch_size
            )
        assert os.path.isfile(input_file_text), f"missing input data, expected at {input_file_text}"
        input_data_text = torch.load(input_file_text)
        output_data_text = inference_text(model, model_name, input_data_text)
        print(f"{output_file_text}")
        torch.save(output_data_text, output_file_text)
    # image
    if overwrite or not image_exists:
        size = model.visual.image_size
        if not isinstance(size, tuple):
            size = (size, size)
        input_file_image = os.path.join(input_dir, f'random_image_{size[0]}_{size[1]}.pt')
        if create_missing_input_data and not os.path.exists(input_file_image):
            create_random_image_data(
                    input_file_image,
                    size,
                    batches = batches,
                    batch_size = batch_size
            )
        assert os.path.isfile(input_file_image), f"missing input data, expected at {input_file_image}"
        input_data_image = torch.load(input_file_image)
        output_data_image = inference_image(model, preprocess_val, input_data_image)
        print(f"{output_file_image}")
        torch.save(output_data_image, output_file_image)

def create_test_data(
        models,
        batches = 1,
        batch_size = 1,
        overwrite = False
):
    models = list(set(models).difference({
            # not available with timm
            # see https://github.com/mlfoundations/open_clip/issues/219
            'timm-convnext_xlarge',
            'timm-vit_medium_patch16_gap_256'
    }).intersection(open_clip.list_models()))
    models.sort()
    print(f"generating test data for:\n{models}")
    for model_name in models:
        print(model_name)
        create_test_data_for_model(
                model_name,
                batches = batches,
                batch_size = batch_size,
                overwrite = overwrite
        )

def main(args):
    import argparse
    parser = argparse.ArgumentParser(description = "Populate test data directory")
    parser.add_argument(
        '-a', '--all',
        action = 'store_true',
        help = "create test data for all models"
    )
    parser.add_argument(
        '-m', '--model',
        type = str,
        default = [],
        nargs = '+',
        help = "model(s) to create test data for"
    )
    parser.add_argument(
        '-f', '--model_file',
        type = str,
        help = "path to a text file containing a list of model names, one line per model"
    )
    parser.add_argument(
        '--overwrite',
        action = 'store_true',
        help = "overwrite existing output data"
    )
    parser.add_argument(
        '-n', '--num_batches',
        default = 1,
        type = int,
        help = "amount of data batches to create (default: 1)"
    )
    parser.add_argument(
        '-b', '--batch_size',
        default = 1,
        type = int,
        help = "test data batch size (default: 1)"
    )
    args = parser.parse_args(args)
    model_file_list = []
    if args.model_file is not None:
        with open(args.model_file, 'r') as f:
            model_file_list = f.read().splitlines()
    if not args.all and len(args.model) < 1 and len(model_file_list) < 1:
        parser.print_help()
        parser.exit(1)
    models = open_clip.list_models() if args.all else args.model + model_file_list
    create_test_data(
        models,
        batches = args.num_batches,
        batch_size = args.batch_size,
        overwrite = args.overwrite
    )


if __name__ == '__main__':
    import sys
    main(sys.argv[1:])

