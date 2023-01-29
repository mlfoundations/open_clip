import os
import random
import numpy as np
from PIL import Image
import torch

if __name__ != '__main__':
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
    
def forward_model(model, model_name, preprocess_val, image_batch, text_batch):
    y = []
    tokenizer = open_clip.get_tokenizer(model_name)
    with torch.no_grad():
        for x_im, x_txt in zip(image_batch, text_batch):
            x_im = torch.stack([preprocess_val(im) for im in x_im])
            x_txt = tokenizer(x_txt)
        y.append(model(x_im, x_txt))
    if type(y[0]) == dict:
        out = {}
        for key in y[0].keys():
            out[key] = torch.stack([batch_out[key] for batch_out in y])
    else:
        out = []
        for i in range(len(y[0])):
            out.append(torch.stack([batch_out[i] for batch_out in y]))
    return out

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
    return models

def _sytem_assert(string):
    assert os.system(string) == 0

class TestWrapper(torch.nn.Module):
    output_dict: torch.jit.Final[bool]
    def __init__(self, model, model_name, output_dict=True) -> None:
        super().__init__()
        self.model = model
        self.output_dict = output_dict
        if type(model) in [open_clip.CLIP, open_clip.CustomTextCLIP]:
            self.model.output_dict = self.output_dict
        config = open_clip.get_model_config(model_name)
        self.head = torch.nn.Linear(config["embed_dim"], 2)

    def forward(self, image, text):
        x = self.model(image, text)
        if self.output_dict:
            out = self.head(x["image_features"])
        else:
            out = self.head(x[0])
        return {"test_output": out}

def main(args):
    global open_clip
    import importlib
    import shutil
    import subprocess
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
        '-f', '--model_list',
        type = str,
        help = "path to a text file containing a list of model names, one model per line"
    )
    parser.add_argument(
        '-s', '--save_model_list',
        type = str,
        help = "path to save the list of models that data was generated for"
    )
    parser.add_argument(
        '-g', '--git_revision',
        type = str,
        help = "git revision to generate test data for"
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
    model_list = []
    if args.model_list is not None:
        with open(args.model_list, 'r') as f:
            model_list = f.read().splitlines()
    if not args.all and len(args.model) < 1 and len(model_list) < 1:
        print("error: at least one model name is required")
        parser.print_help()
        parser.exit(1)
    if args.git_revision is not None:
        stash_output = subprocess.check_output(['git', 'stash']).decode().splitlines()
        has_stash = len(stash_output) > 0 and stash_output[0] != 'No local changes to save'
        current_branch = subprocess.check_output(['git', 'branch', '--show-current'])
        if len(current_branch) < 1:
            # not on a branch -> detached head
            current_branch = subprocess.check_output(['git', 'rev-parse', 'HEAD'])
        current_branch = current_branch.splitlines()[0].decode()
        try:
            _sytem_assert(f'git checkout {args.git_revision}')
        except AssertionError as e:
            _sytem_assert(f'git checkout -f {current_branch}')
            if has_stash:
                os.system(f'git stash pop')
            raise e
    open_clip = importlib.import_module('open_clip')
    models = open_clip.list_models() if args.all else args.model + model_list
    try:
        models = create_test_data(
            models,
            batches = args.num_batches,
            batch_size = args.batch_size,
            overwrite = args.overwrite
        )
    finally:
        if args.git_revision is not None:
            test_dir = os.path.join(os.path.dirname(__file__), 'data')
            test_dir_ref = os.path.join(os.path.dirname(__file__), 'data_ref')
            if os.path.exists(test_dir_ref):
                shutil.rmtree(test_dir_ref, ignore_errors = True)
            if os.path.exists(test_dir):
                os.rename(test_dir, test_dir_ref)
            _sytem_assert(f'git checkout {current_branch}')
            if has_stash:
                os.system(f'git stash pop')
            os.rename(test_dir_ref, test_dir)
    if args.save_model_list is not None:
        print(f"Saving model list as {args.save_model_list}")
        with open(args.save_model_list, 'w') as f:
            for m in models:
                print(m, file=f)


if __name__ == '__main__':
    import sys
    main(sys.argv[1:])

