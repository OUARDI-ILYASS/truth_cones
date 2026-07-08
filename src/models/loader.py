"""nnsight LanguageModel loading with optional 4-bit quantization.

Loads instruction-tuned models via nnsight (§4.1). The paper tests six
models from three families (Tab. master):
    Qwen-2.5 1.5B/7B/14B, Gemma-2 2B/9B, Llama-3.1-8B


"""

from __future__ import annotations

import configparser
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
from nnsight import LanguageModel
from transformers import BitsAndBytesConfig
from huggingface_hub import login
import os

from src.logging_setup import get_logger

logger = get_logger(__name__)


@dataclass
class ModelHandle:
    """Bundle of (LanguageModel, target_layer) returned by load_model.

    target_layer (l* from §3.2) is read from the layer_selection.json
    handoff file by the calling script and attached here so it travels
    with the model through the experiment loop.
    """

    model: LanguageModel
    name: str
    target_layer: Optional[int] = None   # l* from activation patching (§3.2)
    n_layers: int = 0                    # L (total transformer layers)
    hidden_size: int = 0                 # d (residual stream dimension)


def _read_models_ini(ini_path: Path = Path("config.ini")) -> configparser.ConfigParser:
    """Load config.ini if present.

    Falls back to using model_name as a raw HuggingFace ID when no INI
    file is found.
    """
    config = configparser.ConfigParser()
    if ini_path.exists():
        config.read(ini_path)
    return config


def load_model(
    model_name: str,
    quantization: bool = False,
    target_layer: Optional[int] = None,
    ini_path: Path = Path("config.ini"),
) -> ModelHandle:
    """Load a frozen nnsight LanguageModel.

    Args:
        model_name:   key in config.ini (e.g. "qwen-2.5-7b-instruct")
                      or a raw HuggingFace ID.
        quantization: if True, load in 4-bit via bitsandbytes.
        target_layer: l* from activation patching (§3.2), attached to
                      the returned handle.
        ini_path:     path to config.ini.

    Returns:
        ModelHandle with the model, L, d, and l*.
    """
    # ── Authentication and config ─────────────────────────────────────────────
    HF_TOKEN = os.environ.get('RUNPOD_HF_TOKEN')
    login(token=HF_TOKEN)
    config = _read_models_ini(ini_path)
    if model_name in config:
        weights_dir = config[model_name]["weights_directory"]
    else:
        weights_dir = model_name  # raw HuggingFace ID

    logger.info("Loading %s from %s (quantization=%s)", model_name, weights_dir, quantization)


    is_gemma = "gemma" in model_name.lower() or "gemma" in str(weights_dir).lower()

    model_kwargs: dict = {"device_map": {"": 0}}
    if is_gemma:
        model_kwargs["attn_implementation"] = "eager"


    if quantization:
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        model = LanguageModel(
            weights_dir,
            quantization_config=quant_config,
            **model_kwargs,
        )
    else:
        # Gemma must run in bf16: fp16 overflows the soft-cap, fp32
        # wastes memory. Other models default to their config dtype.
        if is_gemma:
            model_kwargs["torch_dtype"] = torch.bfloat16
        model = LanguageModel(weights_dir, **model_kwargs)

    # nnsight tracing requires a pad token.
    if model.tokenizer.pad_token is None:
        model.tokenizer.pad_token = model.tokenizer.eos_token
        model.tokenizer.padding_side = "left"

    n_layers = model.config.num_hidden_layers
    hidden_size = model.config.hidden_size

    logger.info("  n_layers=%d  hidden_size=%d  target_layer=%s",
                n_layers, hidden_size, target_layer)

    return ModelHandle(
        model=model,
        name=model_name,
        target_layer=target_layer,
        n_layers=n_layers,
        hidden_size=hidden_size,
    )