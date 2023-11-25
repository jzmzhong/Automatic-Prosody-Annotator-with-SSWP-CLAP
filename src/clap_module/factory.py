import json
import logging
import os
import re
from copy import deepcopy
from pathlib import Path

import torch

from .model import CLAP, convert_weights_to_fp16

_MODEL_CONFIG_PATHS = [Path(__file__).parent / f"model_configs/"]
_MODEL_CONFIGS = {}  # directory (model_name: config) of model architecture configs


def _natural_key(string_):
    return [int(s) if s.isdigit() else s for s in re.split(r"(\d+)", string_.lower())]


def _rescan_model_configs():
    """Rescan model config directory for new configs.
    """
    global _MODEL_CONFIGS

    config_ext = (".json",)
    config_files = []
    for config_path in _MODEL_CONFIG_PATHS:
        if config_path.is_file() and config_path.suffix in config_ext:
            config_files.append(config_path)
        elif config_path.is_dir():
            for ext in config_ext:
                config_files.extend(config_path.glob(f"*{ext}"))

    for cf in config_files:
        with open(cf, "r") as f:
            model_cfg = json.load(f)
            if all(a in model_cfg for a in ("joint_embed_shape", "audio_cfg", "text_cfg")):
                _MODEL_CONFIGS[cf.stem] = model_cfg

    _MODEL_CONFIGS = {
        k: v
        for k, v in sorted(_MODEL_CONFIGS.items(), key=lambda x: _natural_key(x[0]))
    }


_rescan_model_configs()  # initial populate of model config registry


def load_state_dict(checkpoint_path: str, map_location="cpu", skip_params=True):
    """Load a checkpoint from a file.

    Args:
        checkpoint_path (str): checkpoint path
        map_location (str, optional):  a function, :class:`torch.device`, string or a dict specifying how to
            remap storage locations. Defaults to "cpu".
        skip_params (bool, optional): Remove the module from the key field. Defaults to True.

    Returns:
        state_dict (dict): model state dict

    """
    checkpoint = torch.load(checkpoint_path, map_location=map_location)
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint
    if skip_params:
        if next(iter(state_dict.items()))[0].startswith("module"):
            state_dict = {k[7:]: v for k, v in state_dict.items()}
    return state_dict


def create_model(
        args,
        model_name: str,
        precision: str = "fp32",
        device: torch.device = torch.device("cpu"),
        jit: bool = False,
):
    """Create a CLAP model from a model config.

    Args:
        args (argparse.Namespace): Command-line arguments.
        model_name (str): model name
        precision (str, optional): Model parameter accuracy. Defaults to "fp32".
        device (torch.device, optional): device. Defaults to torch.device("cpu").
        jit (bool, optional): torch.jit.script operations. Defaults to False.

    Returns:
        model (nn.Module): CLAP model
        model_cfg (dict): model config

    """
    if model_name in _MODEL_CONFIGS:
        logging.info(f"Loading {model_name} model config.")
        model_cfg = deepcopy(_MODEL_CONFIGS[model_name])
    else:
        logging.error(
            f"Model config for {model_name} not found; available models {list_models()}."
        )
        raise RuntimeError(f"Model config for {model_name} not found.")
    model = CLAP(args, **model_cfg)

    # load pretrained CLAP model
    if args.pretrained:
        pretrained_clap = torch.load(args.pretrained, map_location='cpu')
        model.load_state_dict(pretrained_clap["state_dict"], strict=False)
        logging.info(f"Loaded pretrained CLAP model weights !!!")
    else:
        # load pretrained audio encoder
        pretrained_audio = model_cfg["audio_cfg"]["pretrained_audio"]
        amodel_type = model_cfg["audio_cfg"]["model_type"]
        if pretrained_audio:
            if amodel_type.startswith('PANN'):
                if 'Cnn14_mAP' in pretrained_audio:  # official checkpoint
                    audio_ckpt = torch.load(pretrained_audio, map_location='cpu')
                    audio_ckpt = audio_ckpt['model']
                    keys = list(audio_ckpt.keys())
                    for key in keys:
                        if 'spectrogram_extractor' not in key and 'logmel_extractor' not in key:
                            v = audio_ckpt.pop(key)
                            audio_ckpt['audio_branch.' + key] = v
                # checkpoint trained via HTSAT codebase
                elif os.path.basename(pretrained_audio).startswith('PANN'):
                    audio_ckpt = torch.load(pretrained_audio, map_location='cpu')
                    audio_ckpt = audio_ckpt['state_dict']
                    keys = list(audio_ckpt.keys())
                    for key in keys:
                        if key.startswith('sed_model'):
                            v = audio_ckpt.pop(key)
                            audio_ckpt['audio_branch.' + key[10:]] = v
                elif os.path.basename(pretrained_audio).startswith(
                        'finetuned'):  # checkpoint trained via linear probe codebase
                    audio_ckpt = torch.load(pretrained_audio, map_location='cpu')
                else:
                    raise ValueError('Unknown audio checkpoint')
            elif amodel_type.startswith('HTSAT'):
                if 'HTSAT_AudioSet_Saved' in pretrained_audio:  # official checkpoint
                    audio_ckpt = torch.load(pretrained_audio, map_location='cpu')
                    audio_ckpt = audio_ckpt['state_dict']
                    keys = list(audio_ckpt.keys())
                    for key in keys:
                        if key.startswith('sed_model') and ('spectrogram_extractor' not in key
                                                            and 'logmel_extractor' not in key):
                            v = audio_ckpt.pop(key)
                            audio_ckpt['audio_branch.' + key[10:]] = v
                # checkpoint trained via HTSAT codebase
                elif os.path.basename(pretrained_audio).startswith('HTSAT'):
                    audio_ckpt = torch.load(pretrained_audio, map_location='cpu')
                    audio_ckpt = audio_ckpt['state_dict']
                    keys = list(audio_ckpt.keys())
                    for key in keys:
                        if key.startswith('sed_model'):
                            v = audio_ckpt.pop(key)
                            audio_ckpt['audio_branch.' + key[10:]] = v
                elif os.path.basename(pretrained_audio).startswith(
                        'finetuned'):  # checkpoint trained via linear probe codebase
                    audio_ckpt = torch.load(pretrained_audio, map_location='cpu')
                else:
                    raise ValueError('Unknown audio checkpoint')
            else:
                raise f'this audio encoder pretrained checkpoint is not support'

            model.load_state_dict(audio_ckpt, strict=False)
            logging.info(f"Loading pretrained {amodel_type} weights ({pretrained_audio}).")
            param_names = [n for n, p in model.named_parameters()]
            for n in param_names:
                print(n, "\t", "Loaded" if n in audio_ckpt else "Unloaded")

    model.to(device=device)
    if precision == "fp16":
        assert device.type != "cpu"
        convert_weights_to_fp16(model)

    if jit:
        model = torch.jit.script(model)

    return model, model_cfg


def list_models():
    """enumerate available model architectures based on config files

    Returns:
        (list) : model config keys
    """
    return list(_MODEL_CONFIGS.keys())


def add_model_config(path):
    """add model config path or file and update registry

    Args:
        path (str): model config path
    """
    if not isinstance(path, Path):
        path = Path(path)
    _MODEL_CONFIG_PATHS.append(path)
    _rescan_model_configs()
