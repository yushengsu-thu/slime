#!/usr/bin/env python
"""Validate the FSDP-to-HF export path using a tiny model checkpoint."""
from __future__ import annotations

import argparse
import os
import shutil
import sys
import tempfile
from typing import Tuple

import torch
import torch.distributed.checkpoint as dist_cp
from transformers import AutoModelForCausalLM, AutoTokenizer
import subprocess


def prepare_checkpoint(model_id: str, work_dir: str) -> Tuple[str, str, str, AutoTokenizer]:
    origin_dir = os.path.join(work_dir, "origin")
    checkpoint_dir = os.path.join(work_dir, "fsdp_ckpt")
    output_dir = os.path.join(work_dir, "converted")

    for path in (origin_dir, checkpoint_dir, output_dir):
        shutil.rmtree(path, ignore_errors=True)
        os.makedirs(path, exist_ok=True)

    model = AutoModelForCausalLM.from_pretrained(model_id)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model.save_pretrained(origin_dir, safe_serialization=True)
    tokenizer.save_pretrained(origin_dir)

    state = {f"model.{k}": v for k, v in model.state_dict().items()}
    dist_cp.save_state_dict(state, storage_writer=dist_cp.FileSystemWriter(checkpoint_dir), no_dist=True)

    return origin_dir, checkpoint_dir, output_dir, tokenizer


def convert_checkpoint(script: str, origin_dir: str, checkpoint_dir: str, output_dir: str) -> None:
    cmd = [
        os.fspath(os.environ.get("PYTHON", sys.executable)),
        script,
        "--input-dir",
        checkpoint_dir,
        "--output-dir",
        output_dir,
        "--origin-hf-dir",
        origin_dir,
        "--force",
    ]
    print(f"Running {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def compare_models(origin_dir: str, converted_dir: str, tokenizer: AutoTokenizer, tolerance: float) -> None:
    base_model = AutoModelForCausalLM.from_pretrained(origin_dir)
    converted_model = AutoModelForCausalLM.from_pretrained(converted_dir)

    inputs = tokenizer("fsdp export smoke test", return_tensors="pt")
    with torch.no_grad():
        base_logits = base_model(**inputs).logits
        converted_logits = converted_model(**inputs).logits

    max_diff = (base_logits - converted_logits).abs().max().item()
    print(f"Max logit delta: {max_diff}")
    if max_diff > tolerance:
        raise SystemExit(f"Converted model diverged from original (>{tolerance})")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", default="sshleifer/tiny-gpt2")
    parser.add_argument(
        "--script",
        default=os.path.join("tools", "convert_torch_dist_to_hf.py"),
        help="Path to convert_torch_dist_to_hf.py",
    )
    parser.add_argument("--tolerance", type=float, default=1e-5, help="Max allowed difference between model logits.")
    parser.add_argument(
        "--work-dir",
        default=None,
        help="Where to place temporary files. Defaults to a temporary directory that is cleaned up at exit.",
    )
    args = parser.parse_args()

    if args.work_dir is None:
        work_dir_obj = tempfile.TemporaryDirectory(prefix="fsdp_verify_")
        work_dir = work_dir_obj.name
    else:
        work_dir_obj = None
        work_dir = args.work_dir
        os.makedirs(work_dir, exist_ok=True)

    origin_dir, checkpoint_dir, output_dir, tokenizer = prepare_checkpoint(args.model_id, work_dir)
    convert_checkpoint(args.script, origin_dir, checkpoint_dir, output_dir)
    compare_models(origin_dir, output_dir, tokenizer, args.tolerance)

    print(f"Verification succeeded. Converted model is saved at {output_dir}")
    if work_dir_obj is not None:
        print(f"Temporary files are kept in {work_dir}")


if __name__ == "__main__":
    main()
