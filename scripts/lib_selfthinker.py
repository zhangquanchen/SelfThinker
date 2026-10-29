"""Shared paths and model loaders for SelfThinker."""
from __future__ import annotations

import os
import sys
from pathlib import Path

CODE_DIR = Path(__file__).resolve().parents[1]
ROOT = (
    Path(os.environ["SELFTHINKER_ROOT"])
    if "SELFTHINKER_ROOT" in os.environ
    else (
        CODE_DIR
        if (CODE_DIR / "models").is_dir() or (CODE_DIR / "data").is_dir()
        else CODE_DIR.parent
    )
)
VISUAL_OPSD_HOME = Path(
    os.environ.get("VISUAL_OPSD_HOME", CODE_DIR / "third_party" / "visual_opsd")
)
THINKMORPH_CKPT = Path(
    os.environ.get("THINKMORPH_MODEL", ROOT / "external" / "ThinkMorph-7B")
)
BAGEL_CKPT = Path(
    os.environ.get("BAGEL_MODEL", ROOT / "external" / "BAGEL-7B-MoT")
)
DATA_ROOT = Path(
    os.environ.get("VISUAL_OPSD_DATA_ROOT", ROOT / "data")
)

os.environ.setdefault("VISUAL_OPSD_DATA_ROOT", str(DATA_ROOT))


def add_sys_path() -> None:
    root = str(VISUAL_OPSD_HOME)
    if root not in sys.path:
        sys.path.insert(0, root)
    scripts_pkg = VISUAL_OPSD_HOME / "scripts"
    init_a = scripts_pkg / "__init__.py"
    init_b = scripts_pkg / "visual_opsd" / "__init__.py"
    for p in (init_a, init_b):
        if not p.exists():
            p.write_text("")


def ckpt_weight_file(model_path: Path | None = None) -> Path:
    model_path = Path(model_path or THINKMORPH_CKPT)
    for name in ("model.safetensors", "ema.safetensors"):
        p = model_path / name
        if p.is_file() and p.stat().st_size > 1_000_000_000:
            return p
    raise FileNotFoundError(
        f"No full weight file under {model_path} "
        f"(looked for model.safetensors / ema.safetensors > 1GB)"
    )


def load_tokenizer(model_path: Path | None = None):
    add_sys_path()
    from data.data_utils import add_special_tokens
    from modeling.qwen2 import Qwen2Tokenizer

    model_path = Path(model_path or THINKMORPH_CKPT)
    tokenizer = Qwen2Tokenizer.from_pretrained(str(model_path))
    tokenizer, new_token_ids, num_new = add_special_tokens(tokenizer)
    return tokenizer, new_token_ids, num_new


def _build_empty_bagel(model_path: Path, visual_gen: bool):
    add_sys_path()
    import torch
    from modeling.autoencoder import load_ae
    from modeling.bagel import (
        Bagel,
        BagelConfig,
        Qwen2Config,
        Qwen2ForCausalLM,
        SiglipVisionConfig,
        SiglipVisionModel,
    )

    llm_config = Qwen2Config.from_json_file(str(model_path / "llm_config.json"))
    llm_config.qk_norm = True
    llm_config.tie_word_embeddings = False
    llm_config.layer_module = "Qwen2MoTDecoderLayer"
    llm_config.freeze_und = False

    vit_config = SiglipVisionConfig.from_json_file(str(model_path / "vit_config.json"))
    vit_config.rope = False
    vit_config.num_hidden_layers = vit_config.num_hidden_layers - 1

    vae_model, vae_config = (None, None)
    if visual_gen:
        vae_model, vae_config = load_ae(local_path=str(model_path / "ae.safetensors"))

    config = BagelConfig(
        visual_gen=visual_gen,
        visual_und=True,
        llm_config=llm_config,
        vit_config=vit_config,
        vae_config=vae_config,
        vit_max_num_patch_per_side=70,
        connector_act="gelu_pytorch_tanh",
        latent_patch_size=2,
        max_latent_size=64,
        interpolate_pos=False,
        timestep_shift=1.0,
    )
    prev = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    try:
        language_model = Qwen2ForCausalLM(llm_config)
        vit_model = SiglipVisionModel(vit_config)
        model = Bagel(language_model, vit_model, config)
        model.vit_model.vision_model.embeddings.convert_conv2d_to_linear(vit_config)
    finally:
        torch.set_default_dtype(prev)
    return model, vae_model, llm_config, vit_config


def load_und_model(model_path: Path | None = None, device: str = "cuda:0"):
    """Load understanding-only pathway (visual_gen=False) for intrinsic u(v) scoring."""
    import torch
    from safetensors.torch import load_file

    model_path = Path(model_path or THINKMORPH_CKPT)
    tokenizer, new_token_ids, num_new = load_tokenizer(model_path)
    model, _, _, _ = _build_empty_bagel(model_path, visual_gen=False)
    if num_new > 0:
        model.language_model.resize_token_embeddings(len(tokenizer))
        model.config.llm_config.vocab_size = len(tokenizer)
        model.language_model.config.vocab_size = len(tokenizer)

    weight = ckpt_weight_file(model_path)
    print(f"[selfthinker] loading understanding weights {weight} -> cpu", flush=True)
    state = load_file(str(weight), device="cpu")
    state.pop("latent_pos_embed.pos_embed", None)
    state.pop("vit_pos_embed.pos_embed", None)
    msg = model.load_state_dict(state, strict=False)
    del state
    print(f"[selfthinker] moving understanding model -> {device}", flush=True)
    model = model.to(device).to(torch.bfloat16).eval()
    return {
        "model": model,
        "tokenizer": tokenizer,
        "new_token_ids": new_token_ids,
        "weight_file": str(weight),
        "missing": list(msg.missing_keys)[:20],
        "unexpected": list(msg.unexpected_keys)[:20],
    }


def load_gen_inferencer(model_path: Path | None = None, max_mem_gib: int = 70):
    """Load full interleaved generator (visual_gen=True) for candidate thought sampling."""
    import torch
    from accelerate import init_empty_weights, load_checkpoint_and_dispatch

    add_sys_path()
    from data.data_utils import add_special_tokens
    from data.transforms import ImageTransform
    from inferencer import InterleaveInferencer
    from modeling.autoencoder import load_ae
    from modeling.bagel import (
        Bagel,
        BagelConfig,
        Qwen2Config,
        Qwen2ForCausalLM,
        SiglipVisionConfig,
        SiglipVisionModel,
    )
    from modeling.qwen2 import Qwen2Tokenizer

    _ = max_mem_gib
    model_path = Path(model_path or THINKMORPH_CKPT)
    llm_config = Qwen2Config.from_json_file(str(model_path / "llm_config.json"))
    llm_config.qk_norm = True
    llm_config.tie_word_embeddings = False
    llm_config.layer_module = "Qwen2MoTDecoderLayer"

    vit_config = SiglipVisionConfig.from_json_file(str(model_path / "vit_config.json"))
    vit_config.rope = False
    vit_config.num_hidden_layers = vit_config.num_hidden_layers - 1

    vae_model, vae_config = load_ae(local_path=str(model_path / "ae.safetensors"))
    config = BagelConfig(
        visual_gen=True,
        visual_und=True,
        llm_config=llm_config,
        vit_config=vit_config,
        vae_config=vae_config,
        vit_max_num_patch_per_side=70,
        connector_act="gelu_pytorch_tanh",
        latent_patch_size=2,
        max_latent_size=64,
    )
    with init_empty_weights():
        language_model = Qwen2ForCausalLM(llm_config)
        vit_model = SiglipVisionModel(vit_config)
        model = Bagel(language_model, vit_model, config)
        model.vit_model.vision_model.embeddings.convert_conv2d_to_linear(
            vit_config, meta=True
        )

    tokenizer = Qwen2Tokenizer.from_pretrained(str(model_path))
    tokenizer, new_token_ids, _ = add_special_tokens(tokenizer)

    weight = ckpt_weight_file(model_path)
    model = load_checkpoint_and_dispatch(
        model,
        checkpoint=str(weight),
        device_map={"": "cuda:0"},
        offload_buffers=False,
        dtype=torch.bfloat16,
        force_hooks=True,
        offload_folder=os.environ.get("SELFTHINKER_OFFLOAD", "/tmp/selfthinker_offload"),
    ).eval()
    vae_model = vae_model.to("cuda:0").to(torch.bfloat16).eval()

    inferencer = InterleaveInferencer(
        model=model,
        vae_model=vae_model,
        tokenizer=tokenizer,
        vae_transform=ImageTransform(1024, 512, 16),
        vit_transform=ImageTransform(980, 224, 14),
        new_token_ids=new_token_ids,
    )
    return {
        "model": model,
        "inferencer": inferencer,
        "tokenizer": tokenizer,
        "new_token_ids": new_token_ids,
        "vae_model": vae_model,
        "weight_file": str(weight),
    }
