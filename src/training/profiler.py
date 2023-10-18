import argparse

import torch
import open_clip
import pandas as pd
from fvcore.nn import FlopCountAnalysis, flop_count_str, ActivationCountAnalysis


parser = argparse.ArgumentParser(description='OpenCLIP Profiler')

# benchmark specific args
parser.add_argument('--model', metavar='NAME', default='',
                    help='model(s) to profile')
parser.add_argument('--results-file', default='', type=str, metavar='FILENAME',
                    help='Output csv file for results')


def profile_fvcore(
        model,
        image_input_size=(3, 224, 224),
        text_input_size=(77,),
        batch_size=1,
        detailed=False,
        force_cpu=False
):
    if force_cpu:
        model = model.to('cpu')
    device, dtype = next(model.parameters()).device, next(model.parameters()).dtype
    example_image_input = torch.ones((batch_size,) + image_input_size, device=device, dtype=dtype)
    example_text_input = torch.ones((batch_size,) + text_input_size, device=device, dtype=torch.int64)
    fca = FlopCountAnalysis(model, (example_image_input, example_text_input))
    aca = ActivationCountAnalysis(model, (example_image_input, example_text_input))
    if detailed:
        fcs = flop_count_str(fca)
        print(fcs)
    return fca.total(), aca.total()


def profile_fvcore_text(
        model,
        text_input_size=(77,),
        batch_size=1,
        detailed=False,
        force_cpu=False
):
    if force_cpu:
        model = model.to('cpu')
    device = next(model.parameters()).device
    example_input = torch.ones((batch_size,) + text_input_size, device=device, dtype=torch.int64)
    fca = FlopCountAnalysis(model, example_input)
    aca = ActivationCountAnalysis(model, example_input)
    if detailed:
        fcs = flop_count_str(fca)
        print(fcs)
    return fca.total(), aca.total()


def profile_fvcore_image(
        model,
        image_input_size=(3, 224, 224),
        batch_size=1,
        detailed=False,
        force_cpu=False
):
    if force_cpu:
        model = model.to('cpu')
    device, dtype = next(model.parameters()).device, next(model.parameters()).dtype
    example_input = torch.ones((batch_size,) + image_input_size, device=device, dtype=dtype)
    fca = FlopCountAnalysis(model, example_input)
    aca = ActivationCountAnalysis(model, example_input)
    if detailed:
        fcs = flop_count_str(fca)
        print(fcs)
    return fca.total(), aca.total()


def count_params(model):
    return sum([m.numel() for m in model.parameters()])


def profile_model(model_name):
    model = open_clip.create_model(model_name, force_custom_text=True, pretrained_hf=False)
    model.eval()
    if torch.cuda.is_available():
        model = model.cuda()

    if isinstance(model.visual.image_size, (tuple, list)):
        image_input_size = (3,) + tuple(model.visual.image_size[-2:])
    else:
        image_input_size = (3, model.visual.image_size, model.visual.image_size)
    text_input_size = (77,)

    results = {}
    results['model'] = model_name
    results['image_size'] = image_input_size[1]

    model_cfg = open_clip.get_model_config(model_name)
    if model_cfg:
        vision_cfg = open_clip.CLIPVisionCfg(**model_cfg['vision_cfg'])
        text_cfg = open_clip.CLIPTextCfg(**model_cfg['text_cfg'])
        results['image_width'] = int(vision_cfg.width)
        results['text_width'] = int(text_cfg.width)
        results['embed_dim'] = int(model_cfg['embed_dim'])
    else:
        results['image_width'] = 0
        results['text_width'] = 0
        results['embed_dim'] = 0

    retries = 2
    while retries:
        retries -= 1
        try:
            macs, acts = profile_fvcore(
                model, image_input_size=image_input_size, text_input_size=text_input_size, force_cpu=not retries)

            image_macs, image_acts = profile_fvcore_image(
                model.visual, image_input_size=image_input_size, force_cpu=not retries)

            text_macs, text_acts = profile_fvcore_text(
                model.text, text_input_size=text_input_size, force_cpu=not retries)

            results['gmacs'] = round(macs / 1e9, 2)
            results['macts'] = round(acts / 1e6, 2)
            results['mparams'] = round(count_params(model) / 1e6, 2)
            results['image_gmacs'] = round(image_macs / 1e9, 2)
            results['image_macts'] = round(image_acts / 1e6, 2)
            results['image_mparams'] = round(count_params(model.visual) / 1e6, 2)
            results['text_gmacs'] = round(text_macs / 1e9, 2)
            results['text_macts'] = round(text_acts / 1e6, 2)
            results['text_mparams'] = round(count_params(model.text) / 1e6, 2)
        except RuntimeError as e:
            pass
    return results


def main():
    args = parser.parse_args()

    # FIXME accept a text file name to allow lists of models in txt/csv
    if args.model == 'all':
        parsed_model = open_clip.list_models()
    else:
        parsed_model = args.model.split(',')

    results = []
    for m in parsed_model:
        row = profile_model(m)
        results.append(row)

    df = pd.DataFrame(results, columns=results[0].keys())
    df = df.sort_values('gmacs')
    print(df)
    if args.results_file:
        df.to_csv(args.results_file, index=False)


if __name__ == '__main__':
    main()
