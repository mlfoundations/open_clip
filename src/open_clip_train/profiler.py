import argparse

import torch
import open_clip
import pandas as pd
from torch.utils.flop_counter import FlopCounterMode
try:
    import fvcore
    import fvcore.nn
except:
    fvcore = None

parser = argparse.ArgumentParser(description='OpenCLIP Profiler')

# benchmark specific args
parser.add_argument('--model', metavar='NAME', default='',
                    help='model(s) to profile')
parser.add_argument('--results-file', default='', type=str, metavar='FILENAME',
                    help='Output csv file for results')
parser.add_argument('--profiler', default='torch', type=str, choices=['torch', 'fvcore'])
parser.add_argument('--batch-size', default=1, type=int, help='Batch size for profiling')
parser.add_argument(
    "--device", default="cuda", type=str, help="Accelerator to use."
)

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
    fca = fvcore.nn.FlopCountAnalysis(model, (example_image_input, example_text_input))
    aca = fvcore.nn.ActivationCountAnalysis(model, (example_image_input, example_text_input))
    if detailed:
        fcs = fvcore.nn.flop_count_str(fca)
        print(fcs)
    return fca.total() / batch_size, aca.total() / batch_size


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
    fca = fvcore.nn.FlopCountAnalysis(model, example_input)
    aca = fvcore.nn.ActivationCountAnalysis(model, example_input)
    if detailed:
        fcs = fvcore.nn.flop_count_str(fca)
        print(fcs)
    return fca.total() / batch_size, aca.total() / batch_size


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
    fca = fvcore.nn.FlopCountAnalysis(model, example_input)
    aca = fvcore.nn.ActivationCountAnalysis(model, example_input)
    if detailed:
        fcs = fvcore.nn.flop_count_str(fca)
        print(fcs)
    return fca.total() / batch_size, aca.total() / batch_size


def profile_torch_image(model, image_input_size, batch_size=1, force_cpu=False):
    """Profile the image encoder using torch.utils.flop_counter"""
    if force_cpu:
        model = model.to('cpu')
    device, dtype = next(model.parameters()).device, next(model.parameters()).dtype
    example_input = torch.ones((batch_size,) + image_input_size, device=device, dtype=dtype)

    flop_counter = FlopCounterMode()
    with flop_counter:
        model(example_input)
    total_flops = sum(flop_counter.get_flop_counts()['Global'].values())
    return total_flops / batch_size


def profile_torch_text(model, text_input_size, batch_size=1, force_cpu=False):
    """Profile the text encoder using torch.utils.flop_counter"""
    if force_cpu:
        model = model.to('cpu')
    device = next(model.parameters()).device
    example_input = torch.ones((batch_size,) + text_input_size, device=device, dtype=torch.int64)

    flop_counter = FlopCounterMode()
    with flop_counter:
        model(example_input)
    total_flops = sum(flop_counter.get_flop_counts()['Global'].values())
    return total_flops / batch_size


def profile_torch(model, text_input_size, image_input_size, batch_size=1, force_cpu=False):
    """Profile the full model using torch.utils.flop_counter"""
    if force_cpu:
        model = model.to('cpu')
    device, dtype = next(model.parameters()).device, next(model.parameters()).dtype
    image_input = torch.ones((batch_size,) + image_input_size, device=device, dtype=dtype)
    text_input = torch.ones((batch_size,) + text_input_size, device=device, dtype=torch.int64)

    flop_counter = FlopCounterMode()
    with flop_counter:
        model(image_input, text_input)
    total_flops = sum(flop_counter.get_flop_counts()['Global'].values())
    return total_flops / batch_size


def count_params(model):
    return sum(m.numel() for m in model.parameters())


def _is_audio_model(model_name):
    """Check if model is a CLAP (audio-text) model."""
    return model_name.startswith("HTSAT") or model_name.startswith("CLAP-HTSAT") or model_name.startswith("whisper")


def profile_torch_audio(model, audio_input, batch_size=1, force_cpu=False):
    """Profile the audio encoder using torch.utils.flop_counter.
    audio_input is a dict with 'waveform' key, as expected by HTSAT.
    """
    if force_cpu:
        model = model.to('cpu')
        audio_input = {k: v.to('cpu') if isinstance(v, torch.Tensor) else v for k, v in audio_input.items()}

    flop_counter = FlopCounterMode()
    with flop_counter:
        model(audio_input)
    total_flops = sum(flop_counter.get_flop_counts()['Global'].values())
    return total_flops / batch_size


def profile_torch_clap(model, audio_input, text_input_size, batch_size=1, force_cpu=False):
    """Profile the full CLAP model using torch.utils.flop_counter."""
    if force_cpu:
        model = model.to('cpu')
    device = next(model.parameters()).device
    text_input = torch.ones((batch_size,) + text_input_size, device=device, dtype=torch.int64)
    audio_input = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in audio_input.items()}

    flop_counter = FlopCounterMode()
    with flop_counter:
        model(audio_input, text_input)
    total_flops = sum(flop_counter.get_flop_counts()['Global'].values())
    return total_flops / batch_size


def profile_model(model_name, batch_size=1, profiler='torch', device="cuda"):
    assert profiler in ['torch', 'fvcore'], 'Only torch and fvcore profilers are supported'
    if profiler == 'fvcore':
        assert fvcore is not None, 'Please install fvcore.'

    is_audio = _is_audio_model(model_name)
    model = open_clip.create_model(model_name, force_custom_text=True, pretrained_hf=False)
    model.eval()

    if torch.cuda.is_available():
        model = model.cuda()
    elif device == "npu" and torch.npu.is_available():
        model = model.npu()

    model_cfg = open_clip.get_model_config(model_name)
    results = {}
    results['model'] = model_name

    text_input_size = (77,)
    if hasattr(model, 'context_length') and model.context_length:
        text_input_size = (model.context_length,)

    if is_audio:
        # Audio model: use audio_cfg for input shape
        audio_cfg = model_cfg.get('audio_cfg', {})
        clip_samples = audio_cfg.get('clip_samples', 480000)
        results['image_size'] = 0
        results['audio_clip_samples'] = clip_samples
        results['audio_sample_rate'] = audio_cfg.get('sample_rate', 48000)
        results['audio_model_type'] = audio_cfg.get('model_type', 'unknown')
        results['audio_model_name'] = audio_cfg.get('model_name', 'unknown')

        text_cfg = model_cfg.get('text_cfg', {})
        results['image_width'] = 0
        results['text_width'] = int(text_cfg.get('width', 0))
        results['embed_dim'] = int(model_cfg.get('embed_dim', 0))

        # Prepare audio input dict as expected by HTSAT
        dev = next(model.parameters()).device
        audio_input = {
            "waveform": torch.randn(batch_size, clip_samples, device=dev),
            "longer": torch.zeros(batch_size, dtype=torch.bool, device=dev),
        }
    else:
        # Vision model: use vision_cfg for input shape
        if isinstance(model.visual.image_size, (tuple, list)):
            image_input_size = (3,) + tuple(model.visual.image_size[-2:])
        else:
            image_input_size = (3, model.visual.image_size, model.visual.image_size)

        results['image_size'] = image_input_size[1]

        if model_cfg and 'vision_cfg' in model_cfg:
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
            results['mparams'] = round(count_params(model) / 1e6, 2)
            if is_audio:
                results['audio_mparams'] = round(count_params(model.audio) / 1e6, 2)
            else:
                results['image_mparams'] = round(count_params(model.visual) / 1e6, 2)
            results['text_mparams'] = round(count_params(model.text) / 1e6, 2)

            if profiler == 'fvcore':
                if is_audio:
                    print(f'  Note: fvcore profiling not yet supported for audio models, skipping FLOPs.')
                else:
                    macs, acts = profile_fvcore(
                        model, image_input_size=image_input_size, text_input_size=text_input_size, force_cpu=not retries, batch_size=batch_size)

                    image_macs, image_acts = profile_fvcore_image(
                        model.visual, image_input_size=image_input_size, force_cpu=not retries, batch_size=batch_size)

                    text_macs, text_acts = profile_fvcore_text(
                        model.text, text_input_size=text_input_size, force_cpu=not retries, batch_size=batch_size)

                    results['gmacs'] = round(macs / 1e9, 2)
                    results['macts'] = round(acts / 1e6, 2)
                    results['image_gmacs'] = round(image_macs / 1e9, 2)
                    results['image_macts'] = round(image_acts / 1e6, 2)
                    results['text_gmacs'] = round(text_macs / 1e9, 2)
                    results['text_macts'] = round(text_acts / 1e6, 2)

            elif profiler == 'torch':
                if is_audio:
                    audio_flops = profile_torch_audio(
                        model.audio, audio_input, force_cpu=not retries, batch_size=batch_size)
                    text_flops = profile_torch_text(
                        model.text, text_input_size=text_input_size, force_cpu=not retries, batch_size=batch_size)
                    total_flops = profile_torch_clap(
                        model, audio_input, text_input_size=text_input_size, force_cpu=not retries, batch_size=batch_size)

                    results['gflops'] = round(total_flops / 1e9, 2)
                    results['audio_gflops'] = round(audio_flops / 1e9, 2)
                    results['text_gflops'] = round(text_flops / 1e9, 2)
                else:
                    image_flops = profile_torch_image(
                        model.visual, image_input_size=image_input_size, force_cpu=not retries, batch_size=batch_size)
                    text_flops = profile_torch_text(
                        model.text, text_input_size=text_input_size, force_cpu=not retries, batch_size=batch_size)
                    total_flops = profile_torch(
                        model, image_input_size=image_input_size, text_input_size=text_input_size, force_cpu=not retries, batch_size=batch_size)

                    results['gflops'] = round(total_flops / 1e9, 2)
                    results['image_gflops'] = round(image_flops / 1e9, 2)
                    results['text_gflops'] = round(text_flops / 1e9, 2)

            break  # success on GPU — don't fall through to CPU which undercounts SDPA attention FLOPs
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
    models_with_errors = []
    for m in parsed_model:
        print('='*100)
        print(f'Profiling {m}')
        try:
            row = profile_model(m, batch_size=args.batch_size, profiler=args.profiler, device=args.device)
            results.append(row)
        except Exception as e:
            print(f'Error profiling {m}: {e}')
            import traceback
            traceback.print_exc()
            models_with_errors.append(m)

    df = pd.DataFrame(results, columns=results[0].keys())

    if 'gmacs' in df.columns:
        df = df.sort_values(by=['gmacs', 'mparams', 'model'])
    else:
        df = df.sort_values(by=['gflops', 'mparams', 'model'])

    print('='*100)
    print('Done.')
    print(df)
    if args.results_file:
        df.to_csv(args.results_file, index=False)

    if models_with_errors:
        print('Models with errors:', models_with_errors)


if __name__ == '__main__':
    main()
