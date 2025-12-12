#!/usr/bin/env python
"""Reproduce issue #1094 by running FSDP conversion without Megatron installed.

Before the fix, importing ``convert_torch_dist_to_hf.py`` failed with
``ModuleNotFoundError: megatron`` even when converting an FSDP checkpoint.
This script prepares a tiny checkpoint and exercises the CLI. On patched
versions the conversion completes successfully; on older versions it will
surface the original import error.
"""
from __future__ import annotations

import argparse
import os
import sys
import shutil
import subprocess
import tempfile
from typing import Tuple

import torch.distributed.checkpoint as dist_cp
from transformers import AutoModelForCausalLM, AutoTokenizer


def prepare_checkpoint(model_id: str, work_dir: str) -> Tuple[str, str, str]:
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

    return origin_dir, checkpoint_dir, output_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", default="sshleifer/tiny-gpt2", help="Hugging Face model id to mirror for the repro.")
    parser.add_argument(
        "--work-dir",
        default=None,
        help="Where to place temporary files. Defaults to a temporary directory that is cleaned up at exit.",
    )
    parser.add_argument(
        "--script",
        default=os.path.join("tools", "convert_torch_dist_to_hf.py"),
        help="Path to convert_torch_dist_to_hf.py (unpatched version will raise ModuleNotFoundError).",
    )
    args = parser.parse_args()

    if args.work_dir is None:
        work_dir_obj = tempfile.TemporaryDirectory(prefix="fsdp_repro_")
        work_dir = work_dir_obj.name
    else:
        work_dir_obj = None
        work_dir = args.work_dir
        os.makedirs(work_dir, exist_ok=True)

    origin_dir, checkpoint_dir, output_dir = prepare_checkpoint(args.model_id, work_dir)

    print(f"Prepared FSDP checkpoint under {checkpoint_dir}")
    print("Running conversion script (older versions fail with ModuleNotFoundError: megatron)...")

    cmd = [
        os.fspath(os.environ.get("PYTHON", sys.executable)),
        args.script,
        "--input-dir",
        checkpoint_dir,
        "--output-dir",
        output_dir,
        "--origin-hf-dir",
        origin_dir,
        "--force",
    ]
    subprocess.run(cmd, check=True)
    print(f"Conversion finished; Hugging Face weights are in {output_dir}")

    if work_dir_obj is not None:
        print(f"Temporary files are kept in {work_dir}")


if __name__ == "__main__":
    main()
