import json
import logging
import os
import re
import warnings
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import torch

from .convert import convert_state_dict
from .model import CLIP, CustomTextCLIP, convert_weights_to_lp, convert_to_custom_text_state_dict,\
    resize_pos_embed, get_cast_dtype, resize_text_pos_embed, set_model_preprocess_cfg
from .coca_model import CoCa
from .loss import ClipLoss, DistillClipLoss, CoCaLoss, SigLipLoss
from .pretrained import is_pretrained_cfg, get_pretrained_cfg, download_pretrained,\
    list_pretrained_tags_by_model, download_pretrained_from_hf
from .transform import image_transform_v2, AugmentationCfg, PreprocessCfg, merge_preprocess_dict, merge_preprocess_kwargs
from .tokenizer import HFTokenizer, SimpleTokenizer, SigLipTokenizer, DEFAULT_CONTEXT_LENGTH

HF_HUB_PREFIX = 'hf-hub:'
_MODEL_CONFIG_PATHS = [Path(__file__).parent / f"model_configs/"]
_MODEL_CONFIGS = {}  # directory (model_name: config) of model architecture configs


def _natural_key(string_):
    return [int(s) if s.isdigit() else s for s in re.split(r'(\d+)', string_.lower())]


def _rescan_model_configs():
    global _MODEL_CONFIGS

    config_ext = ('.json',)
    config_files = []
    for config_path in _MODEL_CONFIG_PATHS:
        if config_path.is_file() and config_path.suffix in config_ext:
            config_files.append(config_path)
        elif config_path.is_dir():
            for ext in config_ext:
                config_files.extend(config_path.glob(f'*{ext}'))

    for cf in config_files:
        with open(cf, 'r') as f:
            model_cfg = json.load(f)
            if all(a in model_cfg for a in ('embed_dim', 'vision_cfg', 'text_cfg')):
                _MODEL_CONFIGS[cf.stem] = model_cfg

    _MODEL_CONFIGS = {k: v for k, v in sorted(_MODEL_CONFIGS.items(), key=lambda x: _natural_key(x[0]))}


_rescan_model_configs()  # initial populate of model config registry


def list_models():
    """ enumerate available model architectures based on config files """
    return list(_MODEL_CONFIGS.keys())


def add_model_config(path):
    """ add model config path or file and update registry """
    if not isinstance(path, Path):
        path = Path(path)
    _MODEL_CONFIG_PATHS.append(path)
    _rescan_model_configs()


# Define Schema Prefixes as constants
HF_HUB_PREFIX = 'hf-hub:'
LOCAL_DIR_PREFIX = 'local-dir:'

def parse_model_name(model_name: str) -> Tuple[Optional[str], str]:
    """
    Parses a model name string to identify a schema and the remaining identifier.

    Args:
        model_name: The model name string (e.g., 'ViT-B-32',
                    'hf-hub:org/repo', 'local-dir:/path/to/dir',
                    'local-dir:./relative/path').

    Returns:
        A tuple (schema, identifier):
          - schema (Optional[str]): 'hf-hub', 'local-dir', or None if no schema detected.
          - identifier (str): The part after the schema prefix, or the original
                              string if no schema was present. For 'local-dir',
                              this is the raw path string provided.
    Raises:
        ValueError: If a schema prefix is present but the identifier part is empty.
    """
    # Check for local directory schema first
    if model_name.startswith(LOCAL_DIR_PREFIX):
        # Extract the identifier (path) after the prefix
        identifier = model_name[len(LOCAL_DIR_PREFIX):]
        # Validate that the identifier (path) is not empty
        if not identifier:
            raise ValueError("Empty path specified after 'local-dir:' schema.")
        # Return the schema and the raw path identifier
        # Note: We don't resolve or fully validate the path here,
        #       that's left to the calling function (e.g., using os.path.isdir)
        return 'local-dir', identifier

    # Check for Hugging Face Hub schema
    elif model_name.startswith(HF_HUB_PREFIX):
        # Extract the identifier (HF Hub ID) after the prefix
        identifier = model_name[len(HF_HUB_PREFIX):]
        # Validate that the identifier is not empty
        if not identifier:
            raise ValueError("Empty identifier specified after 'hf-hub:' schema.")
        # Return the schema and the HF Hub ID
        return 'hf-hub', identifier

    # If neither schema prefix is found
    else:
        # No schema detected, return None for schema and the original string as identifier
        return None, model_name


def _get_hf_config(
        model_id: str,
        cache_dir: Optional[str] = None,
):
    """ Fetch model config from HuggingFace Hub.
    """
    config_path = download_pretrained_from_hf(
        model_id,
        filename='open_clip_config.json',
        cache_dir=cache_dir,
    )
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    return config


def get_model_config(model_name):
    """ Fetch model config from schema specified location or local library configs.
    """
    loc, model_id = parse_model_name(model_name)
    if loc == 'local-dir':
        local_path = Path(model_id) / 'open_clip_config.json'
        with open(local_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        return config.get('model_cfg', config)
    elif loc == 'hf-hub':
        config = _get_hf_config(model_id)
        return config.get('model_cfg', config)
    elif model_name in _MODEL_CONFIGS:
        return deepcopy(_MODEL_CONFIGS[model_name])
    else:
        return None


def load_state_dict(
        checkpoint_path: str,
        device='cpu',
        weights_only=True,
):
    # Check if safetensors or not and load weights accordingly
    if str(checkpoint_path).endswith(".safetensors"):
        from safetensors.torch import load_file
        checkpoint = load_file(checkpoint_path, device=device)
    else:
        try:
            checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=weights_only)
        except TypeError:
            checkpoint = torch.load(checkpoint_path, map_location=device)

    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    elif isinstance(checkpoint, torch.jit.ScriptModule):
        state_dict = checkpoint.state_dict()
        for key in ["input_resolution", "context_length", "vocab_size"]:
            state_dict.pop(key, None)
    else:
        state_dict = checkpoint
    if next(iter(state_dict.items()))[0].startswith('module'):
        state_dict = {k[7:]: v for k, v in state_dict.items()}
    return state_dict


def load_checkpoint(
        model: Union[CLIP, CustomTextCLIP],
        checkpoint_path: str,
        strict: bool = True,
        weights_only: bool = True,
        device='cpu',
):
    if Path(checkpoint_path).suffix in ('.npz', '.npy'):
        # Separate path loading numpy big_vision (SigLIP) weights
        from open_clip.convert import load_big_vision_weights
        load_big_vision_weights(model, checkpoint_path)
        return {}

    state_dict = load_state_dict(checkpoint_path, device=device, weights_only=weights_only)

    # Detect & convert 3rd party state_dicts -> open_clip
    state_dict = convert_state_dict(model, state_dict)

    # Detect old format and make compatible with new format
    if 'positional_embedding' in state_dict and not hasattr(model, 'positional_embedding'):
        state_dict = convert_to_custom_text_state_dict(state_dict)

    # correct if logit_scale differs in being scaler vs 1d param
    if 'logit_scale' in state_dict and model.logit_scale.ndim != state_dict['logit_scale'].ndim:
        state_dict['logit_scale'] = state_dict['logit_scale'].reshape(model.logit_scale.shape)

    # correct if logit_bias differs in being scaler vs 1d param
    if 'logit_bias' in state_dict and model.logit_bias.ndim != state_dict['logit_bias'].ndim:
        state_dict['logit_bias'] = state_dict['logit_bias'].reshape(model.logit_bias.shape)

    # If loading a non-SigLIP model for SigLIP training. See https://github.com/mlfoundations/open_clip/issues/712
    if 'logit_bias' not in state_dict and model.logit_bias is not None:
        state_dict["logit_bias"] = torch.zeros_like(state_dict["logit_scale"])

    # Certain text transformers no longer expect position_ids after transformers==4.31
    position_id_key = 'text.transformer.embeddings.position_ids'
    if position_id_key in state_dict and not hasattr(model, position_id_key):
        del state_dict[position_id_key]

    resize_pos_embed(state_dict, model)
    resize_text_pos_embed(state_dict, model)

    # Finally, load the massaged state_dict into model
    incompatible_keys = model.load_state_dict(state_dict, strict=strict)
    return incompatible_keys


def _find_checkpoint_in_dir(dir_path: Path) -> Optional[str]:
    checkpoints = list(dir_path.glob('*.safetensors')) + list(dir_path.glob('*.bin')) + list(dir_path.glob('*.pth'))
    if not checkpoints:
        return None
    checkpoints.sort()
    checkpoints.sort(key=lambda x: x.suffix == '.safetensors', reverse=True)
    preferred_order = [
        "open_clip_model.safetensors", "open_clip_pytorch_model.safetensors",
        "open_clip_pytorch_model.bin",  "open_clip_pytorch_model.pth",
        "model.safetensors", "pytorch_model.bin", "pytorch_model.pth", "model.pth"
    ]
    preferred_checkpoints = [c for c in checkpoints if c.name in preferred_order]
    if preferred_checkpoints:
        preferred_checkpoints.sort(key=lambda x: preferred_order.index(x.name))
        chosen = preferred_checkpoints[0]
        logging.info(f"Found preferred checkpoint file: {chosen.name} in {dir_path}")
        return str(chosen)
    chosen = checkpoints[0]
    logging.warning(
        f"Multiple checkpoints found in {dir_path}: {[c.name for c in checkpoints]}. Using '{chosen.name}'.")
    return str(chosen)


def create_model(
        model_name: str, # Can contain schemas 'hf-hub:' or 'local-dir:'
        pretrained: Optional[str] = None, # Used ONLY if model_name has NO schema
        load_weights: bool = True,
        precision: str = 'fp32',
        device: Union[str, torch.device] = 'cpu',
        jit: bool = False,
        force_quick_gelu: bool = False,
        force_custom_text: bool = False,
        force_patch_dropout: Optional[float] = None,
        force_image_size: Optional[Union[int, Tuple[int, int]]] = None,
        force_preprocess_cfg: Optional[Dict[str, Any]] = None,
        force_context_length: Optional[int] = None,
        pretrained_image: bool = False, # Load default base image weights (at creation, if no CLIP weights)
        pretrained_text: bool = True,  # Load default base text weights (at creation, if no CLIP weights) - NEW
        pretrained_image_path: Optional[str] = None, # Load specific image weights from file (after creation)
        pretrained_text_path: Optional[str] = None, # Load specific text weights from file (after creation)
        cache_dir: Optional[str] = None,
        output_dict: Optional[bool] = None,
        require_pretrained: bool = False,
        weights_only: bool = True,
        **model_kwargs,
) -> torch.nn.Module:
    """
    Creates and configures a contrastive vision-language model.

    `model_name` specifies architecture/config source:
      - 'ViT-B-32': Built-in model name. `pretrained` specifies CLIP weights source (tag or file path).
      - 'hf-hub:org/repo': Loads config/weights from HF Hub. `pretrained` is IGNORED.
      - 'local-dir:/path/to/folder': Loads config/weights from local dir. `pretrained` is IGNORED.

    Base tower weights loading controlled by `pretrained_image` and `pretrained_text` flags,
    only effective if no full CLIP checkpoint (`pretrained` or schema source) is loaded.

    Tower-specific weights can be loaded *after* creation via `pretrained_image_path`
    and `pretrained_text_path`.

    Args:
        model_name: Model identifier, potentially with schema ('hf-hub:', 'local-dir:').
        pretrained: Source for CLIP weights (tag or file path) ONLY if model_name has no schema.
        load_weights: Load the resolved pretrained weights if True, otherwise random init or tower overrides only.
        precision: Model precision ('fp32', 'fp16', 'bf16', ...).
        device: Device ('cpu', 'cuda', ...).
        jit: If True, JIT compile the model.
        force_quick_gelu: Force use of QuickGELU activation in model config.
        force_custom_text: Force use of custom text encoder architecture.
        force_patch_dropout: Override patch dropout value in model config.
        force_image_size: Override image size in model config.
        force_preprocess_cfg: Dict to override specific FINAL preprocessing parameters.
        force_context_length: Override context length in model config.
        pretrained_image: Load default base weights for image tower at creation if no CLIP weights loaded.
        pretrained_text: Load default base weights for text tower at creation if no CLIP weights loaded (default: True).
        pretrained_image_path: Path to load weights specifically into image tower after creation.
        pretrained_text_path: Path to load weights specifically into text tower after creation.
        cache_dir: Cache directory for downloads.
        output_dict: If True and model supports it, return dict output.
        require_pretrained: Raise error if no `pretrained` CLIP weights loaded when required.
        weights_only: Use weights_only=True for torch.load (safer).
        **model_kwargs: Additional keyword arguments for model constructor (highest override priority).

    Returns:
        The created model instance.
    """
    schema, identifier = parse_model_name(model_name)
    if 'pretrained_hf' in model_kwargs:
        # for backwards compat, override pretrained_text
        pretrained_text = model_kwargs.pop('pretrained_hf')
    if isinstance(device, str):
        device = torch.device(device)

    model_cfg = None
    preprocess_cfg = asdict(PreprocessCfg())  # Populate with defaults
    checkpoint_path = None # Final path for full CLIP weights
    pretrained_cfg_for_tag = None # Store tag config if pretrained is a tag and schema is None

    logging.info(f"Parsing model identifier. Schema: {schema}, Identifier: {identifier}")
    if schema and pretrained:
        logging.warning(f"Ignoring `pretrained='{pretrained}'` because `model_name` has '{schema}' schema.")
        pretrained = None  # Nullify pretrained as it's ignored

    # Handle schemas first - these ignore the `pretrained` argument
    if schema == 'local-dir':
        # Handle local directory schema
        local_path = Path(identifier)
        if not local_path.is_dir():
            raise FileNotFoundError(f"Directory specified via 'local-dir:' schema not found: {local_path}")

        local_config_path = local_path / 'open_clip_config.json'
        logging.info(f"Attempting to load config from local dir: {local_config_path}")
        if local_config_path.is_file():
            try:
                # Try loading and parsing the JSON config
                with open(local_config_path, 'r', encoding='utf-8') as f:
                    local_json_config = json.load(f)
                # Check if the required 'model_cfg' key is present
                if 'model_cfg' in local_json_config:
                    # Load model config and merge preprocess config
                    model_cfg = local_json_config['model_cfg']
                    preprocess_cfg = merge_preprocess_dict(preprocess_cfg, local_json_config.get('preprocess_cfg', {}))
                    logging.info(f"Loaded model config and preprocess from: {local_config_path}")
                    # Look for weights checkpoint in the same directory
                    checkpoint_path = _find_checkpoint_in_dir(local_path)
                    if checkpoint_path:
                        logging.info(f"Found CLIP weights in local folder: {checkpoint_path}")
                    else:
                        logging.warning(f"Local config loaded, but no CLIP weights found in {local_path}")
                else:
                    # Config file exists but lacks the necessary key
                    raise ValueError(f"Local config {local_config_path} missing 'model_cfg'.")
            except Exception as e:
                # Handle JSON parsing errors or other exceptions during config load
                raise ValueError(f"Could not load valid config from specified 'local-dir:{identifier}': {e}") from e
        else:
            # Directory exists but the config file is missing
            raise FileNotFoundError(f"'local-dir:' specified, but config file missing: {local_config_path}")

    elif schema == 'hf-hub':
        # Handle Hugging Face Hub schema
        model_id = identifier
        logging.info(f"Attempting to load config from HF Hub: {model_id}")
        try:
            # Fetch configuration from Hugging Face Hub
            hf_config = _get_hf_config(model_id, cache_dir=cache_dir)
            if 'model_cfg' not in hf_config:
                raise RuntimeError(f"'model_cfg' not found in config from {model_id}")
            # Load model config and merge preprocess config
            model_cfg = hf_config['model_cfg']
            preprocess_cfg = merge_preprocess_dict(preprocess_cfg, hf_config.get('preprocess_cfg', {}))
            logging.info(f"Loaded model config from HF Hub: {model_id}")
            # Attempt find default weights file from the Hub repo
            try:
                checkpoint_path = download_pretrained_from_hf(model_id, cache_dir=cache_dir)
                logging.info(f"Found default weights file on HF Hub: {checkpoint_path}")
            except Exception as e_weights:
                # Log warning if weights download fails, but proceed (might only need config)
                logging.warning(f"Could not find/download default weights on HF Hub for {model_id}: {e_weights}")
        except Exception as e_config:
            # Handle errors during config fetching from HF Hub
            raise RuntimeError(f"Failed initial config/weights load from HF Hub {model_id}: {e_config}") from e_config

    # No Schema Prefix - Use built-in name + pretrained arg (tag or file)
    elif schema is None:
        # Handle model names without schema prefix
        # Use identifier (original model_name) and clean it for lookup
        model_name_cleaned = identifier.replace('/', '-')

        # Get base config from built-in name using the cleaned identifier
        model_cfg = get_model_config(model_name_cleaned)
        if model_cfg is None:
            # Raise error if no matching built-in config found
            raise RuntimeError(
                f"Model config for '{model_name_cleaned}' not found in built-ins. Available: {list_models()}")
        logging.info(f"Loaded built-in {model_name_cleaned} model config.")

        # Determine checkpoint path and update preprocess_cfg based on `pretrained` arg (tag or file)
        if pretrained:
            # Check if `pretrained` is a known tag
            pretrained_cfg_for_tag = get_pretrained_cfg(model_name_cleaned, pretrained)
            if pretrained_cfg_for_tag:
                try:
                    # Download weights associated with the tag
                    checkpoint_path = download_pretrained(pretrained_cfg_for_tag, cache_dir=cache_dir)
                    preprocess_cfg = merge_preprocess_dict(preprocess_cfg, pretrained_cfg_for_tag)
                    # QuickGELU compatibility check will happen in after force overrides
                except Exception as e:
                    logging.error(f"Failed to download weights for tag '{pretrained}': {e}")
                    raise RuntimeError(f"Failed to download weights for tag '{pretrained}': {e}")
            elif os.path.isfile(pretrained):
                # Handle pretrained file path
                logging.info(f"`pretrained` specifies file path: {pretrained}")
                checkpoint_path = pretrained
            else:
                logging.error(
                    f"Pretrained tag or path ({pretrained}) for '{model_name_cleaned}' not found. "
                    f"Available tags: {list_pretrained_tags_by_model(model_name_cleaned)}"
                )
                raise RuntimeError(f"Pretrained value '{pretrained}' is not a known tag or valid file path")

    # Apply model config overrides
    if model_cfg is None:
        raise RuntimeError("Model configuration could not be determined after Stage 1.")
    text_cfg = model_cfg['text_cfg']
    vision_cfg = model_cfg['vision_cfg']
    if force_quick_gelu:
        model_cfg["quick_gelu"] = True
    if force_patch_dropout is not None:
        vision_cfg["patch_dropout"] = force_patch_dropout
    if force_image_size is not None:
        vision_cfg["image_size"] = force_image_size
    if force_context_length is not None:
        text_cfg["context_length"] = force_context_length

    # Check compatibility (e.g., QuickGELU warning for tags)
    if schema is None and pretrained_cfg_for_tag:
        # Only perform check if config came from built-in and weights from a tag
        model_quick_gelu = model_cfg.get('quick_gelu', False) # Check the potentially overridden value
        tag_quick_gelu = pretrained_cfg_for_tag.get('quick_gelu', False)
        if tag_quick_gelu != model_quick_gelu:
            # Warn if the final model config's GELU setting mismatches the tag's training setting
             warnings.warn(
                 f"QuickGELU mismatch between final model config (quick_gelu={model_quick_gelu}) "
                 f"and pretrained tag '{pretrained}' (quick_gelu={tag_quick_gelu}).",
                 UserWarning
             )

    # Decide whether to use the checkpoint path based on load_weights
    if checkpoint_path is not None:
        if not load_weights:
            logging.info(
                f"Potential checkpoint path '{checkpoint_path}' found, but skipping assignment due to load_weights=False.")
            checkpoint_path = None
    else:
        logging.info("No potential checkpoint path found from config source or pretrained arg.")

    # Set default base weight loading flags for image and text towers
    # Only load base pretrained weights if other weights will not be loaded into respective towers
    enable_default_image_weights = pretrained_image and pretrained_image_path is None and checkpoint_path is None
    enable_default_text_weights = pretrained_text and pretrained_text_path is None and checkpoint_path is None
    is_timm_model = 'timm_model_name' in model_cfg.get("vision_cfg", {})
    is_hf_text_model = 'hf_model_name' in model_cfg.get('text_cfg', {})
    if is_timm_model:
        vision_cfg['timm_model_pretrained'] = enable_default_image_weights
    else:
        enable_default_image_weights = False  # for accurate logging
    if is_hf_text_model:
        text_cfg['hf_model_pretrained'] = enable_default_text_weights
    else:
        enable_default_text_weights = False  # for accurate logging

    # Determine model class (CLIP, CustomTextCLIP, CoCa)
    custom_text = model_cfg.pop('custom_text', False) or force_custom_text or is_hf_text_model
    if custom_text:
        # Use CustomTextCLIP (or CoCa if multimodal_cfg is present)
        if "multimodal_cfg" in model_cfg:
            model_class = CoCa
        else:
            model_class = CustomTextCLIP
    else:
        # Default to standard CLIP
        model_class = CLIP

    # Apply final **kwargs overrides (highest priority) to a copy of model_cfg
    final_model_cfg = deepcopy(model_cfg)
    final_model_cfg.update(model_kwargs)

    # Get casting dtype based on precision argument
    cast_dtype = get_cast_dtype(precision)

    # Instantiate the model
    logging.info(f"Instantiating model architecture: {model_class.__name__}")
    model = model_class(**final_model_cfg, cast_dtype=cast_dtype)
    _set_model_device_and_precision(model, device, precision, is_timm_model)

    # Load Full Pretrained CLIP Weights (if path exists)
    pretrained_loaded = False
    if checkpoint_path:
        logging.info(f'Loading full pretrained weights from: {checkpoint_path}')
        # Use the load_checkpoint helper which handles state dict loading, conversions, etc.
        # Use strict=True by default for full model loading to catch mismatches.
        load_checkpoint(
            model,
            checkpoint_path,
            strict=True,
            weights_only=weights_only,
            device='cpu' # Load to CPU first
        )
        pretrained_loaded = True

    # Load tower-specific weights (image and text), after the full CLIP checkpoint, potentially overwriting parts.
    pretrained_image_loaded = False # Track if specific image weights loaded
    if pretrained_image_path:
        if os.path.isfile(pretrained_image_path):
            logging.info(f"Attempting to load image tower weights from: {pretrained_image_path}")
            try:
                # Load the state dict from the file
                image_state_dict = load_state_dict(
                    pretrained_image_path,
                    device='cpu',
                    weights_only=weights_only
                )
                # Check if model has the 'visual' attribute
                if hasattr(model, 'visual'):
                    # Load into the visual tower, use strict=False for flexibility
                    incompatible_keys = model.visual.load_state_dict(image_state_dict, strict=False)
                    logging.info(
                        f"Loaded image tower weights from {pretrained_image_path}. Incompatible keys: {incompatible_keys}")
                    pretrained_image_loaded = True # Mark specific image weights as loaded
                else:
                    # Model structure doesn't match expectation
                    logging.warning(
                        f"Model does not have a 'visual' attribute, cannot load image tower weights from {pretrained_image_path}")
            except Exception as e:
                # Handle errors during image tower weight loading
                logging.error(f"Error loading image tower weights from {pretrained_image_path}: {e}")
        else:
            # Path provided is not a valid file
            logging.warning(f"Invalid file path specified for pretrained_image_path: {pretrained_image_path}")

    pretrained_text_loaded = False # Track if specific text weights loaded
    if pretrained_text_path:
        if os.path.isfile(pretrained_text_path):
            logging.info(f"Attempting to load text tower weights from: {pretrained_text_path}")
            try:
                # Load the state dict from the file
                text_state_dict = load_state_dict(
                    pretrained_text_path,
                    device='cpu',
                    weights_only=weights_only
                )
                # Safely get the text attribute (usually 'text', but could be different)
                text_module = getattr(model, 'text', model)
                if text_module is not None:
                    # Load into the text tower, use strict=False for flexibility
                    incompatible_keys = text_module.load_state_dict(text_state_dict, strict=False)
                    logging.info(f"Loaded text tower weights from {pretrained_text_path}. Incompatible keys: {incompatible_keys}")
                    pretrained_text_loaded = True # Mark specific text weights as loaded
                else:
                    # Model structure doesn't match expectation
                    logging.warning(f"Model does not have a standard 'text' attribute, cannot load text tower weights from {pretrained_text_path}")
            except Exception as e:
                # Handle errors during text tower weight loading
                logging.error(f"Error loading text tower weights from {pretrained_text_path}: {e}")
        else:
            # Path provided is not a valid file
            logging.warning(f"Invalid file path specified for pretrained_text_path: {pretrained_text_path}")

    partially_loaded = enable_default_text_weights or enable_default_image_weights \
        or pretrained_image_loaded or pretrained_text_loaded
    if require_pretrained and not pretrained_loaded:
         # If CLIP weights were required but failed to load, raise an error.
         # Loading tower-specific weights does not satisfy `require_pretrained`.
         raise RuntimeError(
             f"Required pretrained weights (`model_name='{model_name}', pretrained='{pretrained}'`) could not be loaded. "
         )
    elif not pretrained_loaded and partially_loaded:
         # Some tower weights loaded
         logging.warning(f"Model {model_name} initialized partially.")
    elif not pretrained_loaded and not partially_loaded:
         # Absolutely no weights were loaded from any source
         logging.warning(f"No pretrained weights loaded for model '{model_name}'. Model initialized randomly.")

    if output_dict and hasattr(model, "output_dict"):
        # Enable dictionary output if model supports it
        model.output_dict = True

    # If force_image_size was specified and we have a timm model, call set_input_size after loading weights
    if force_image_size is not None and is_timm_model and hasattr(model.visual, 'set_input_size'):
        logging.info(f"Calling set_input_size({force_image_size}) on timm vision model.")
        model.visual.set_input_size(force_image_size)

    if jit:
        logging.info("Attempting JIT scripting...")
        try:
            model = torch.jit.script(model)
            logging.info("JIT scripting successful.")
        except Exception as e:
            logging.warning(f"JIT scripting failed: {e}. Returning non-JIT model.")

    # Prepare and set final preprocessing configuration on the model
    final_preprocess_cfg = deepcopy(preprocess_cfg) # Start with config determined earlier
    # Ensure image_size in preprocess config matches the actual model's visual component size, if possible
    visual_module = getattr(model, 'visual', None)
    if visual_module is not None and hasattr(visual_module, 'image_size'):
        # Update preprocess size from the instantiated visual module
         final_preprocess_cfg['size'] = visual_module.image_size
    # Apply force_preprocess_cfg overrides (highest priority for preprocessing)
    final_preprocess_cfg = merge_preprocess_dict(final_preprocess_cfg, force_preprocess_cfg or {})

    # Attach the final config to the model
    set_model_preprocess_cfg(model, final_preprocess_cfg)
    logging.info(f"Final image preprocessing configuration set: {final_preprocess_cfg}")

    # Log completion and return the configured model
    logging.info(f"Model {model_name} creation process complete.")
    return model


def get_tokenizer(
        model_name: str = '',
        context_length: Optional[int] = None,
        cache_dir: Optional[str] = None,
        **kwargs, # Additional tokenizer kwargs passed to constructor
):
    """
    Gets the appropriate tokenizer based on the model identifier schema or name.

    `model_name` can specify source via schema:
      - 'ViT-B-32': Looks up built-in config to determine tokenizer type.
      - 'hf-hub:org/repo': Loads config from HF Hub to determine tokenizer type.
      - 'local-dir:/path/to/folder': Loads config from local dir to determine tokenizer type.
    """
    schema, identifier = parse_model_name(model_name)

    config = {} # Stores the loaded model_cfg relevant section (usually text_cfg)
    local_dir_path = None # Store path if schema is local-dir to resolve relative paths
    hf_fallback_id = None

    # Determine Configuration Source based on Schema
    logging.info(f"Parsing tokenizer identifier. Schema: {schema}, Identifier: {identifier}")

    if schema == 'local-dir':
        # Handle local directory schema
        local_dir_path = Path(identifier) # Store the path for later use
        if not local_dir_path.is_dir():
            raise FileNotFoundError(f"Directory specified via 'local-dir:' schema not found at {local_dir_path}")
        local_config_path = local_dir_path / 'open_clip_config.json'
        logging.info(f"Attempting to load config from local-dir: {local_config_path}")
        if local_config_path.is_file():
            try:
                # Load and parse the JSON config
                with open(local_config_path, 'r', encoding='utf-8') as f:
                    local_json_config = json.load(f)
                if 'model_cfg' in local_json_config:
                    config = local_json_config['model_cfg']
                else:
                    raise ValueError(f"Local config {local_config_path} missing 'model_cfg'.")
            except Exception as e:
                raise ValueError(f"Could not load valid config for 'local-dir:{identifier}' ({e}).") from e
        else:
             raise FileNotFoundError(f"'local-dir:' specified, but config file missing: {local_config_path}")

    elif schema == 'hf-hub':
        # Handle Hugging Face Hub schema
        model_id = identifier
        logging.info(f"Attempting to load config from hf-hub:{model_id}")
        config_err = ''
        try:
            # Fetch config from HF Hub
            hf_config = _get_hf_config(model_id, cache_dir=cache_dir)
            config = hf_config.get('model_cfg', None)
            if not config:
                config_err = 'model_cfg key not found'
        except Exception as e:
            config_err = str(e)
        if not config:
            hf_fallback_id = model_id
            config = {}
            logging.warning(
                f"Could not load config from hf-hub:{model_id} ({config_err})."
                f"Falling back to using model_id for tokenizer.")

    elif schema is None and identifier:
        # Try built-in config lookup using the identifier (original model_name)
        logging.info(f"Attempting to load config from built-in: {identifier}")
        config = get_model_config(identifier)

    # Check if config determination failed completely (should only be possible if initial schema parsing failed badly)
    if config is None:
        logging.warning(f"Model configuration not found, returning default SimpleTokenizer.")
        return SimpleTokenizer(context_length=context_length or DEFAULT_CONTEXT_LENGTH, **kwargs)

    # Safely access text_cfg even if config is {} (from non-builtin name case)
    text_config = config.get('text_cfg', {})

    # Resolve context length: argument > config > default
    if context_length is None:
        # Use context_length from text_cfg if available, otherwise default
        context_length = text_config.get('context_length', DEFAULT_CONTEXT_LENGTH)

    # Merge tokenizer kwargs: function kwargs override config kwargs
    tokenizer_kwargs = text_config.get('tokenizer_kwargs', {}) # Start with config kwargs
    tokenizer_kwargs.update(kwargs) # Apply caller kwargs, overriding config ones

    # Get the specified HF tokenizer name from config, if any
    hf_tokenizer_name = text_config.get('hf_tokenizer_name', '')
    if not hf_tokenizer_name and hf_fallback_id:
        hf_tokenizer_name = hf_fallback_id

    if hf_tokenizer_name:
        # If 'hf_tokenizer_name' key exists in text_cfg (even if empty string): Use HFTokenizer.
        if schema == 'local-dir':
            # If config came from local-dir, ALWAYS use the local dir path for HFTokenizer.
            # This assumes the tokenizer files are inside that directory.
            tokenizer_source = local_dir_path
        else:
            tokenizer_source = hf_tokenizer_name
        tokenizer_mode = text_config.get('tokenizer_mode', None)

        logging.info(f"Using HFTokenizer with source: '{tokenizer_source}', mode: '{tokenizer_mode}'")
        tokenizer = HFTokenizer(
            tokenizer_source,
            context_length=context_length,
            cache_dir=cache_dir,
            tokenizer_mode=tokenizer_mode,
            **tokenizer_kwargs,
        )

    elif schema is None and 'siglip' in identifier.lower():
        # Check for SigLIP naming convention ONLY if no schema was present AND no hf_tokenizer_name found
        # Avoids misinterpreting 'local-dir:/path/with/siglip/in/name'
        tn_variant = 'gemma' if 'siglip2' in identifier.lower() else 'mc4' if 'i18n' in identifier.lower() else 'c4-en'
        logging.info(f"Using SigLipTokenizer variant: {tn_variant}")
        tokenizer = SigLipTokenizer(
            tn_variant,
            context_length=context_length,
        )
    else:
        # Default to SimpleTokenizer if no HF specified and not SigLIP name match
        logging.info("Using default SimpleTokenizer.")
        tokenizer = SimpleTokenizer(
            context_length=context_length,
            **tokenizer_kwargs,
        )

    return tokenizer


def _set_model_device_and_precision(
        model: torch.nn.Module,
        device: torch.device,
        precision: str,
        is_timm_model: bool = False
):
    if precision in ("fp16", "bf16"):
        dtype = torch.float16 if 'fp16' in precision else torch.bfloat16
        # manual mixed precision that matches original OpenAI behaviour
        if is_timm_model:
            from .transformer import LayerNormFp32
            # FIXME this is a bit janky, create timm based model in low-precision and
            # then cast only LayerNormFp32 instances back to float32 so they don't break.
            # Why? The convert_weights_to_lp fn only works with native models.
            model.to(device=device, dtype=dtype)

            def _convert_ln(m):
                if isinstance(m, LayerNormFp32):
                    m.weight.data = m.weight.data.to(torch.float32)
                    m.bias.data = m.bias.data.to(torch.float32)

            model.apply(_convert_ln)
        else:
            model.to(device=device)
            convert_weights_to_lp(model, dtype=dtype)
    elif precision in ("pure_fp16", "pure_bf16"):
        dtype = torch.float16 if 'fp16' in precision else torch.bfloat16
        model.to(device=device, dtype=dtype)
    else:
        model.to(device=device)


def create_loss(args):
    if args.distill:
        return DistillClipLoss(
            local_loss=args.local_loss,
            gather_with_grad=args.gather_with_grad,
            cache_labels=True,
            rank=args.rank,
            world_size=args.world_size,
            use_horovod=args.horovod,
        )
    elif "coca" in args.model.lower():
        return CoCaLoss(
            caption_loss_weight=args.coca_caption_loss_weight,
            clip_loss_weight=args.coca_contrastive_loss_weight,
            local_loss=args.local_loss,
            gather_with_grad=args.gather_with_grad,
            cache_labels=True,
            rank=args.rank,
            world_size=args.world_size,
            use_horovod=args.horovod,
        )
    elif args.siglip:
        assert not args.horovod, "Horovod not currently supported for SigLip"
        return SigLipLoss(
            rank=args.rank,
            world_size=args.world_size,
            dist_impl=args.loss_dist_impl,  # siglip has multiple distributed implementations to choose from
        )

    return ClipLoss(
        local_loss=args.local_loss,
        gather_with_grad=args.gather_with_grad,
        cache_labels=True,
        rank=args.rank,
        world_size=args.world_size,
        use_horovod=args.horovod,
    )


def create_model_and_transforms(
        model_name: str,
        pretrained: Optional[str] = None,
        load_weights: bool = True,
        precision: str = 'fp32',
        device: Union[str, torch.device] = 'cpu',
        jit: bool = False,
        force_quick_gelu: bool = False,
        force_custom_text: bool = False,
        force_patch_dropout: Optional[float] = None,
        force_image_size: Optional[Union[int, Tuple[int, int]]] = None,
        force_context_length: Optional[int] = None,
        image_mean: Optional[Tuple[float, ...]] = None,
        image_std: Optional[Tuple[float, ...]] = None,
        image_interpolation: Optional[str] = None,
        image_resize_mode: Optional[str] = None,  # only effective for inference
        aug_cfg: Optional[Union[Dict[str, Any], AugmentationCfg]] = None,
        pretrained_image: bool = False,
        pretrained_text: bool = True,
        pretrained_image_path: Optional[str] = None,
        pretrained_text_path: Optional[str] = None,
        cache_dir: Optional[str] = None,
        output_dict: Optional[bool] = None,
        weights_only: bool = True,
        **model_kwargs,
):
    """
    Creates a contrastive vision-language model along with preprocessing transforms for training and validation.

    This function combines model creation with the generation of appropriate image preprocessing pipelines,
    making it convenient for training workflows where both model and transforms are needed.

    `model_name` specifies architecture/config source:
      - 'ViT-B-32': Built-in model name. `pretrained` specifies CLIP weights source (tag or file path).
      - 'hf-hub:org/repo': Loads config/weights from HF Hub. `pretrained` is IGNORED.
      - 'local-dir:/path/to/folder': Loads config/weights from local dir. `pretrained` is IGNORED.

    The preprocessing transforms are automatically configured based on the model's requirements,
    with separate pipelines for training (with augmentation) and validation (without augmentation).

    Args:
        model_name: Model identifier, potentially with schema ('hf-hub:', 'local-dir:').
        pretrained: Source for CLIP weights (tag or file path) ONLY if model_name has no schema.
        load_weights: Load the resolved pretrained weights if True, otherwise random init or tower overrides only.
        precision: Model precision ('fp32', 'fp16', 'bf16', ...).
        device: Device ('cpu', 'cuda', ...).
        jit: If True, JIT compile the model.
        force_quick_gelu: Force use of QuickGELU activation in model config.
        force_custom_text: Force use of custom text encoder architecture.
        force_patch_dropout: Override patch dropout value in model config.
        force_image_size: Override image size in model config.
        force_context_length: Override context length in model config.
        image_mean: Override default image normalization mean values (per channel).
        image_std: Override default image normalization std values (per channel).
        image_interpolation: Override default interpolation method for image resizing.
        image_resize_mode: Override resize mode for inference preprocessing ('squash', 'longest', 'shortest').
        aug_cfg: Augmentation configuration for training transforms. Can be dict or AugmentationCfg object.
                 Controls random crop, color jitter, etc. If None, uses model defaults.
        pretrained_image: Load default (timm) base weights for image tower at creation if no CLIP weights loaded.
        pretrained_text: Load default (hf) base weights for text tower at creation if no CLIP weights loaded.
        pretrained_image_path: Path to load weights specifically into image tower after creation.
        pretrained_text_path: Path to load weights specifically into text tower after creation.
        cache_dir: Cache directory for downloads.
        output_dict: If True and model supports it, return dict output.
        weights_only: Use weights_only=True for torch.load (safer).
        **model_kwargs: Additional keyword arguments for model constructor (highest override priority).

    Returns:
        Tuple[torch.nn.Module, Callable, Callable]: A tuple containing:
            - model: The created model instance
            - preprocess_train: Image preprocessing transform for training (includes augmentation)
            - preprocess_val: Image preprocessing transform for validation/inference (no augmentation)

    Example:
        >>> # Basic usage with built-in model
        >>> model, train_transform, val_transform = create_model_and_transforms('ViT-B-32', pretrained='openai')
        >>>
        >>> # With custom augmentation
        >>> aug_cfg = {'scale': (0.9, 1.0), 'ratio': (1.0, 1.0)}
        >>> model, train_transform, val_transform = create_model_and_transforms(
        ...     'ViT-L-14',
        ...     pretrained='datacomp_xl_s13b_b90k',
        ...     aug_cfg=aug_cfg
        ... )
        >>>
        >>> # From Hugging Face Hub
        >>> model, train_transform, val_transform = create_model_and_transforms('hf-hub:org/model-repo')

    Note:
        The training transform includes data augmentation based on `aug_cfg`, while the validation
        transform performs only the necessary preprocessing (resize, center crop, normalize) without
        any random augmentation.
    """
    force_preprocess_cfg = merge_preprocess_kwargs(
        {},
        mean=image_mean,
        std=image_std,
        interpolation=image_interpolation,
        resize_mode=image_resize_mode,
    )

    model = create_model(
        model_name,
        pretrained,
        load_weights=load_weights,
        precision=precision,
        device=device,
        jit=jit,
        force_quick_gelu=force_quick_gelu,
        force_custom_text=force_custom_text,
        force_patch_dropout=force_patch_dropout,
        force_image_size=force_image_size,
        force_preprocess_cfg=force_preprocess_cfg,
        force_context_length=force_context_length,
        pretrained_image=pretrained_image,
        pretrained_text=pretrained_text,
        pretrained_image_path=pretrained_image_path,
        pretrained_text_path=pretrained_text_path,
        cache_dir=cache_dir,
        output_dict=output_dict,
        weights_only=weights_only,
        **model_kwargs,
    )

    pp_cfg = PreprocessCfg(**model.visual.preprocess_cfg)

    preprocess_train = image_transform_v2(
        pp_cfg,
        is_train=True,
        aug_cfg=aug_cfg,
    )
    preprocess_val = image_transform_v2(
        pp_cfg,
        is_train=False,
    )

    return model, preprocess_train, preprocess_val


def create_model_from_pretrained(
        model_name: str,
        pretrained: Optional[str] = None,
        precision: str = 'fp32',
        device: Union[str, torch.device] = 'cpu',
        jit: bool = False,
        force_quick_gelu: bool = False,
        force_custom_text: bool = False,
        force_image_size: Optional[Union[int, Tuple[int, int]]] = None,
        force_context_length: Optional[int] = None,
        image_mean: Optional[Tuple[float, ...]] = None,
        image_std: Optional[Tuple[float, ...]] = None,
        image_interpolation: Optional[str] = None,
        image_resize_mode: Optional[str] = None,  # only effective for inference
        return_transform: bool = True,
        cache_dir: Optional[str] = None,
        weights_only: bool = True,
        **model_kwargs,
):
    """
    Creates a contrastive vision-language model from pretrained weights with optional preprocessing transform.

    This function is a convenience wrapper around `create_model` that enforces loading of pretrained weights
    (require_pretrained=True) and optionally returns the appropriate preprocessing transform for inference.
    It's designed for use cases where a pretrained model is required, such as feature extraction,
    zero-shot classification, or fine-tuning.

    `model_name` specifies architecture/config source:
      - 'ViT-B-32': Built-in model name. `pretrained` specifies CLIP weights source (tag or file path).
      - 'hf-hub:org/repo': Loads config/weights from HF Hub. `pretrained` is IGNORED.
      - 'local-dir:/path/to/folder': Loads config/weights from local dir. `pretrained` is IGNORED.

    Unlike `create_model`, this function will raise an error if pretrained weights cannot be loaded.

    Args:
        model_name: Model identifier, potentially with schema ('hf-hub:', 'local-dir:').
        pretrained: Source for CLIP weights (tag or file path) ONLY if model_name has no schema.
                   If None and schema requires it, will raise an error.
        precision: Model precision ('fp32', 'fp16', 'bf16', ...).
        device: Device ('cpu', 'cuda', ...).
        jit: If True, JIT compile the model.
        force_quick_gelu: Force use of QuickGELU activation in model config.
        force_custom_text: Force use of custom text encoder architecture.
        force_image_size: Override image size in model config. Useful for using models at different resolutions.
        force_context_length: Override context length in model config.
        image_mean: Override default image normalization mean values (per channel).
        image_std: Override default image normalization std values (per channel).
        image_interpolation: Override default interpolation method for image resizing ('bicubic', 'bilinear', 'nearest').
        image_resize_mode: Override resize mode for inference preprocessing ('squash', 'longest', 'shortest').
            Only affects the returned preprocessing transform, not training.
        return_transform: If True, returns (model, preprocess). If False, returns only model.
        cache_dir: Cache directory for downloads.
        weights_only: Use weights_only=True for torch.load (safer).
        **model_kwargs: Additional keyword arguments for model constructor (highest override priority).

    Returns:
        Union[torch.nn.Module, Tuple[torch.nn.Module, Callable]]:
            - If return_transform=False: Just the model instance
            - If return_transform=True: Tuple of (model, preprocess) where preprocess is the
              inference preprocessing transform

    Raises:
        RuntimeError: If pretrained weights are required but cannot be loaded.

    Example:
        >>> # Load model with preprocessing
        >>> model, preprocess = create_model_from_pretrained('ViT-B-32', pretrained='openai')
        >>>
        >>> # Load model without preprocessing (e.g., when using custom preprocessing)
        >>> model = create_model_from_pretrained('ViT-B-32', pretrained='openai', return_transform=False)
        >>>
        >>> # Load from Hugging Face Hub
        >>> model, preprocess = create_model_from_pretrained('hf-hub:laion/CLIP-ViT-L-14-DataComp.XL-s13B-b90K')
        >>>
        >>> # Load with custom image size
        >>> model, preprocess = create_model_from_pretrained(
        ...     'ViT-L-14',
        ...     pretrained='openai',
        ...     force_image_size=336
        ... )

    Note:
        This function always requires pretrained weights to be available and loaded successfully.
        For cases where you want to create a model without pretrained weights or with only
        partial weight loading, use `create_model` or `create_model_and_transforms` instead.
    """
    force_preprocess_cfg = merge_preprocess_kwargs(
        {},
        mean=image_mean,
        std=image_std,
        interpolation=image_interpolation,
        resize_mode=image_resize_mode,
    )

    model = create_model(
        model_name,
        pretrained,
        precision=precision,
        device=device,
        jit=jit,
        force_quick_gelu=force_quick_gelu,
        force_custom_text=force_custom_text,
        force_image_size=force_image_size,
        force_preprocess_cfg=force_preprocess_cfg,
        force_context_length=force_context_length,
        cache_dir=cache_dir,
        require_pretrained=True,
        weights_only=weights_only,
        **model_kwargs,
    )

    if not return_transform:
        return model

    preprocess = image_transform_v2(
        PreprocessCfg(**model.visual.preprocess_cfg),
        is_train=False,
    )

    return model, preprocess
