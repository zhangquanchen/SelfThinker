#!/usr/bin/env python3
"""Register SelfThinker model names in a local VLMEvalKit-ThinkMorph checkout."""
from __future__ import annotations

import sys
from pathlib import Path

MARKER = "thinkmorph_series[\"SelfThinker\"]"


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("usage: register_models.py /path/to/VLMEvalKit_Thinkmorph")
    config = Path(sys.argv[1]) / "vlmeval" / "config.py"
    text = config.read_text()
    if "_SELFTHINKER_ROOT" not in text:
        text = text.replace(
            "import os\n",
            "import os\n"
            '_CODE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))\n'
            "_SELFTHINKER_ROOT = os.environ.get(\n"
            '    "SELFTHINKER_ROOT",\n'
            '    _CODE_DIR if (os.path.isdir(os.path.join(_CODE_DIR, "models")) or os.path.isdir(os.path.join(_CODE_DIR, "data"))) else os.path.dirname(_CODE_DIR),\n'
            ")\n",
            1,
        )
    if MARKER not in text:
        block = '''
_SELFTHINKER_MODEL_PATH = os.environ.get(
    "SELFTHINKER_MODEL",
    _SELFTHINKER_ROOT + "/models"
    if os.path.isfile(_SELFTHINKER_ROOT + "/models/model.safetensors")
    else _SELFTHINKER_ROOT + "/models/selfthinker",
)
thinkmorph_series["SelfThinker"] = partial(
    ThinkMorph,
    model_path=_SELFTHINKER_MODEL_PATH,
    think=True,
    understanding_output=True,
    temperature=0.0,
    do_sample=False,
    max_think_token_n=1024,
    save_dir=_SELFTHINKER_ROOT + "/outputs/eval_selfthinker/imgs",
    use_torch_compile=False,
)
thinkmorph_series["ThinkMorph-7B"] = partial(
    ThinkMorph,
    model_path=os.environ.get("THINKMORPH_MODEL", _SELFTHINKER_ROOT + "/external/ThinkMorph-7B"),
    think=True,
    understanding_output=True,
    temperature=0.0,
    do_sample=False,
    max_think_token_n=1024,
    save_dir=_SELFTHINKER_ROOT + "/outputs/eval_thinkmorph/imgs",
    use_torch_compile=False,
)
thinkmorph_series["BAGEL-7B-MoT"] = partial(
    ThinkMorph,
    model_path=os.environ.get("BAGEL_MODEL", _SELFTHINKER_ROOT + "/external/BAGEL-7B-MoT"),
    think=True,
    understanding_output=True,
    temperature=0.0,
    do_sample=False,
    max_think_token_n=1024,
    save_dir=_SELFTHINKER_ROOT + "/outputs/eval_bagel/imgs",
    use_torch_compile=False,
)

'''
        mark = "supported_VLM = {}"
        if mark not in text:
            raise SystemExit(f"could not find {mark} in {config}")
        text = text.replace(mark, block + mark, 1)
    config.write_text(text)
    print(f"registered models in {config}")


if __name__ == "__main__":
    main()
