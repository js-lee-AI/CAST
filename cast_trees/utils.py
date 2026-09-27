import json
import os

import torch


def load_models(target_path, draft_path, device="cuda", dtype=torch.bfloat16, attn="sdpa"):
    from dflash.model import DFlashDraftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    target = AutoModelForCausalLM.from_pretrained(
        target_path, dtype=dtype, attn_implementation=attn).to(device).eval()
    draft = DFlashDraftModel.from_pretrained(draft_path, dtype=dtype).to(device).eval()
    tok = AutoTokenizer.from_pretrained(target_path)
    return target, draft, tok


def env_info():
    import transformers
    cc = torch.cuda.get_device_capability(0)
    return {"gpu": torch.cuda.get_device_name(0), "sm": f"{cc[0]}{cc[1]}",
            "torch": torch.__version__, "cuda": torch.version.cuda,
            "transformers": transformers.__version__}


def dump(obj, path):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)
