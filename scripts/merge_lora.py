"""Merge a PEFT LoRA adapter into its base model and save a standalone checkpoint.

Typical usage (when `lora_path` contains `adapter_config.json`):

  python scripts/merge_lora.py --lora_path /path/to/adapter --out_path /path/to/merged

Optionally override base model path:

  python scripts/merge_lora.py --base_model_path /path/to/base --lora_path /path/to/adapter --out_path /path/to/merged
"""

from __future__ import annotations

import argparse
import os
from typing import Optional


def _normalize_dtype(dtype: str) -> str:
    dtype = dtype.lower()
    if dtype in {"bf16", "bfloat16"}:
        return "bf16"
    if dtype in {"fp16", "float16"}:
        return "fp16"
    if dtype in {"fp32", "float32"}:
        return "fp32"
    raise ValueError(f"Unsupported dtype: {dtype}")


def _default_device_map() -> str:
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--lora_path", required=True, help="Path to LoRA adapter directory")
    p.add_argument("--out_path", required=True, help="Output directory for merged model")
    p.add_argument(
        "--base_model_path",
        default="",
        help="Optional base model path. If empty, will read from adapter config.",
    )
    p.add_argument(
        "--dtype",
        default="bf16",
        choices=["bf16", "fp16", "fp32"],
        help="Model dtype used during merge.",
    )
    p.add_argument(
        "--device_map",
        default="",
        help='Device map passed to HF model loader (e.g., "cuda", "cpu", "auto").',
    )
    p.add_argument(
        "--attn_implementation",
        default="flash_attention_2",
        help="Attention implementation for loading base model.",
    )
    p.add_argument(
        "--trust_remote_code",
        action="store_true",
        default=True,
        help="Trust remote code when loading Qwen models.",
    )
    p.add_argument(
        "--skip_if_exists",
        action="store_true",
        help="Skip merging if out_path already looks like a full model checkpoint.",
    )
    p.add_argument(
        "--dry_run",
        action="store_true",
        help="Only validate paths and print resolved base/adapter/output.",
    )
    return p.parse_args()


def _resolve_base_model_path(lora_path: str, override: str) -> str:
    if override:
        return override
    try:
        from peft import PeftConfig

        peft_config = PeftConfig.from_pretrained(lora_path)
        return peft_config.base_model_name_or_path
    except Exception as e:
        raise RuntimeError(
            "Failed to infer base_model_path from adapter config. "
            "Please pass --base_model_path explicitly. "
            f"Original error: {e}"
        )


def _looks_like_full_model(path: str) -> bool:
    if not os.path.isdir(path):
        return False
    # Transformers weight files
    for fname in (
        "model.safetensors",
        "pytorch_model.bin",
        "model.safetensors.index.json",
        "pytorch_model.bin.index.json",
    ):
        if os.path.exists(os.path.join(path, fname)):
            return True
    return False


def main() -> None:
    args = parse_args()

    if not os.path.isdir(args.lora_path):
        raise FileNotFoundError(f"lora_path not found: {args.lora_path}")

    adapter_cfg = os.path.join(args.lora_path, "adapter_config.json")
    if not os.path.exists(adapter_cfg):
        raise FileNotFoundError(
            f"adapter_config.json not found under lora_path: {args.lora_path}"
        )

    if args.skip_if_exists and _looks_like_full_model(args.out_path):
        print(f"[merge_lora] out_path already contains full model, skip: {args.out_path}")
        return

    base_model_path = _resolve_base_model_path(args.lora_path, args.base_model_path)

    device_map = args.device_map or _default_device_map()
    dtype_key = _normalize_dtype(args.dtype)

    print("[merge_lora] base_model_path:", base_model_path)
    print("[merge_lora] lora_path      :", args.lora_path)
    print("[merge_lora] out_path       :", args.out_path)
    print("[merge_lora] dtype          :", args.dtype)
    print("[merge_lora] device_map     :", device_map)

    if args.dry_run:
        return

    try:
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor
        from peft import PeftModel
    except Exception as e:
        raise RuntimeError(
            "Missing required packages for merging. "
            "Install training requirements (torch/transformers/peft) first. "
            f"Original error: {e}"
        )

    dtype = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }[dtype_key]

    os.makedirs(args.out_path, exist_ok=True)

    base = AutoModelForImageTextToText.from_pretrained(
        base_model_path,
        torch_dtype=dtype,
        attn_implementation=args.attn_implementation,
        device_map=device_map,
        trust_remote_code=args.trust_remote_code,
    ).eval()

    model = PeftModel.from_pretrained(base, args.lora_path).eval()
    model = model.merge_and_unload()
    model.save_pretrained(args.out_path, safe_serialization=True)

    # Prefer loading processor from adapter path (may contain updated templates/special tokens)
    try:
        processor = AutoProcessor.from_pretrained(
            args.lora_path,
            trust_remote_code=args.trust_remote_code,
        )
    except Exception:
        processor = AutoProcessor.from_pretrained(
            base_model_path,
            trust_remote_code=args.trust_remote_code,
        )
    processor.save_pretrained(args.out_path)

    print("[merge_lora] Merged model saved to:", args.out_path)


if __name__ == "__main__":
    main()
