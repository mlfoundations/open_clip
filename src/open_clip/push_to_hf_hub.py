import argparse
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Optional, Tuple

import torch

try:
    from huggingface_hub import (
        create_repo,
        get_hf_file_metadata,
        hf_hub_download,
        hf_hub_url,
        repo_type_and_id_from_hf_id,
        upload_folder,
    )
    from huggingface_hub.utils import EntryNotFoundError
    _has_hf_hub = True
except ImportError:
    _has_hf_hub = False

from .factory import create_model_from_pretrained, get_model_config, get_tokenizer
from .tokenizer import HFTokenizer


def save_config_for_hf(
        model,
        config_path: str,
        model_config: Optional[dict]
):
    preprocess_cfg = {
        'mean': model.visual.image_mean,
        'std': model.visual.image_std,
    }
    hf_config = {
        'model_cfg': model_config,
        'preprocess_cfg': preprocess_cfg,
    }

    with config_path.open('w') as f:
        json.dump(hf_config, f, indent=2)


def save_for_hf(
    model,
    tokenizer: HFTokenizer,
    model_config: dict,
    save_directory: str,
    weights_filename='open_clip_pytorch_model.bin',
    config_filename='open_clip_config.json',
):
    save_directory = Path(save_directory)
    save_directory.mkdir(exist_ok=True, parents=True)

    weights_path = save_directory / weights_filename
    torch.save(model.state_dict(), weights_path)

    tokenizer.save_pretrained(save_directory)

    config_path = save_directory / config_filename
    save_config_for_hf(model, config_path, model_config=model_config)


def push_to_hf_hub(
    model,
    tokenizer,
    model_config: Optional[dict],
    repo_id: str,
    commit_message: str = 'Add model',
    token: Optional[str] = None,
    revision: Optional[str] = None,
    private: bool = False,
    create_pr: bool = False,
    model_card: Optional[dict] = None,
):
    if not isinstance(tokenizer, HFTokenizer):
        # default CLIP tokenizers use https://huggingface.co/openai/clip-vit-large-patch14
        tokenizer = HFTokenizer('openai/clip-vit-large-patch14')

    # Create repo if it doesn't exist yet
    repo_url = create_repo(repo_id, token=token, private=private, exist_ok=True)

    # Infer complete repo_id from repo_url
    # Can be different from the input `repo_id` if repo_owner was implicit
    _, repo_owner, repo_name = repo_type_and_id_from_hf_id(repo_url)
    repo_id = f"{repo_owner}/{repo_name}"

    # Check if README file already exist in repo
    try:
        get_hf_file_metadata(hf_hub_url(repo_id=repo_id, filename="README.md", revision=revision))
        has_readme = True
    except EntryNotFoundError:
        has_readme = False

    # Dump model and push to Hub
    with TemporaryDirectory() as tmpdir:
        # Save model weights and config.
        save_for_hf(
            model,
            tokenizer=tokenizer,
            model_config=model_config,
            save_directory=tmpdir,
        )

        # Add readme if it does not exist
        if not has_readme:
            model_card = model_card or {}
            model_name = repo_id.split('/')[-1]
            readme_path = Path(tmpdir) / "README.md"
            readme_text = generate_readme(model_card, model_name)
            readme_path.write_text(readme_text)

        # Upload model and return
        return upload_folder(
            repo_id=repo_id,
            folder_path=tmpdir,
            revision=revision,
            create_pr=create_pr,
            commit_message=commit_message,
        )


def push_pretrained_to_hf_hub(
    model_name,
    pretrained: str,
    repo_id: str,
    image_mean: Optional[Tuple[float, ...]] = None,
    image_std: Optional[Tuple[float, ...]] = None,
    commit_message: str = 'Add model',
    token: Optional[str] = None,
    revision: Optional[str] = None,
    private: bool = False,
    create_pr: bool = False,
    model_card: Optional[dict] = None,
):
    model, preprocess_eval = create_model_from_pretrained(
        model_name,
        pretrained=pretrained,
        image_mean=image_mean,
        image_std=image_std,
    )

    model_config = get_model_config(model_name)
    assert model_config

    tokenizer = get_tokenizer(model_name)

    push_to_hf_hub(
        model=model,
        tokenizer=tokenizer,
        model_config=model_config,
        repo_id=repo_id,
        commit_message=commit_message,
        token=token,
        revision=revision,
        private=private,
        create_pr=create_pr,
        model_card=model_card,
    )


def generate_readme(model_card: dict, model_name: str):
    readme_text = "---\n"
    readme_text += "tags:\n- zero-shot-image-classification\n- clip\n"
    readme_text += "library_tag: open_clip\n"
    readme_text += f"license: {model_card.get('license', 'mit')}\n"
    if 'details' in model_card and 'Dataset' in model_card['details']:
        readme_text += 'datasets:\n'
        readme_text += f"- {model_card['details']['Dataset'].lower()}\n"
    readme_text += "---\n"
    readme_text += f"# Model card for {model_name}\n"
    if 'description' in model_card:
        readme_text += f"\n{model_card['description']}\n"
    if 'details' in model_card:
        readme_text += f"\n## Model Details\n"
        for k, v in model_card['details'].items():
            if isinstance(v, (list, tuple)):
                readme_text += f"- **{k}:**\n"
                for vi in v:
                    readme_text += f"  - {vi}\n"
            elif isinstance(v, dict):
                readme_text += f"- **{k}:**\n"
                for ki, vi in v.items():
                    readme_text += f"  - {ki}: {vi}\n"
            else:
                readme_text += f"- **{k}:** {v}\n"
    if 'usage' in model_card:
        readme_text += f"\n## Model Usage\n"
        readme_text += model_card['usage']
        readme_text += '\n'

    if 'comparison' in model_card:
        readme_text += f"\n## Model Comparison\n"
        readme_text += model_card['comparison']
        readme_text += '\n'

    if 'citation' in model_card:
        readme_text += f"\n## Citation\n"
        if not isinstance(model_card['citation'], (list, tuple)):
            citations = [model_card['citation']]
        else:
            citations = model_card['citation']
        for c in citations:
            readme_text += f"```bibtex\n{c}\n```\n"

    return readme_text


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Push to Hugging Face Hub")
    parser.add_argument(
        "--model", type=str, help="Name of the model to use.",
    )
    parser.add_argument(
        "--pretrained", type=str,
        help="Use a pretrained CLIP model weights with the specified tag or file path.",
    )
    parser.add_argument(
        "--repo-id", type=str,
        help="Destination HF Hub repo-id ie 'organization/model_id'.",
    )
    parser.add_argument(
        '--image-mean', type=float, nargs='+', default=None, metavar='MEAN',
        help='Override default image mean value of dataset')
    parser.add_argument(
        '--image-std', type=float, nargs='+', default=None, metavar='STD',
        help='Override default image std deviation of of dataset')
    args = parser.parse_args()

    print(f'Saving model {args.model} with pretrained weights {args.pretrained} to Hugging Face Hub at {args.repo_id}')

    # FIXME add support to pass model_card json / template from file via cmd line

    push_pretrained_to_hf_hub(
        args.model,
        args.pretrained,
        args.repo_id,
        image_mean=args.image_mean,  # override image mean/std if trained w/ non defaults
        image_std=args.image_std,
    )

    print(f'{args.model} saved.')
