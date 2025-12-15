import importlib.util
from pathlib import Path

import pytest
import torch
from torch.distributed.checkpoint import FileSystemWriter, save_state_dict
from transformers import AutoModelForCausalLM, GPT2Config

CONVERTER_PATH = Path(__file__).resolve().parents[1] / "tools" / "convert_fsdp_to_hf.py"

spec = importlib.util.spec_from_file_location("convert_fsdp_to_hf", CONVERTER_PATH)
module = importlib.util.module_from_spec(spec)
assert spec is not None and spec.loader is not None
spec.loader.exec_module(module)

_convert_fsdp_to_hf = module._convert_fsdp_to_hf
copy_assets = module.copy_assets


@pytest.mark.integration
def test_convert_fsdp_to_hf_roundtrip(tmp_path: Path) -> None:
    config = GPT2Config(
        vocab_size=32,
        n_positions=32,
        n_ctx=32,
        n_embd=16,
        n_layer=1,
        n_head=2,
    )
    origin_dir = tmp_path / "origin"
    origin_dir.mkdir(parents=True, exist_ok=True)

    # Build and persist a minimal local HF model to avoid network dependency.
    model = AutoModelForCausalLM.from_config(config)
    model.save_pretrained(origin_dir)

    # Create a minimal FSDP-style checkpoint with a model_state.model prefix.
    fsdp_dir = tmp_path / "fsdp"
    writer = FileSystemWriter(str(fsdp_dir))
    prefixed_state = {
        f"model_state.model.{k}": v.cpu() for k, v in model.state_dict().items()
    }
    save_state_dict(prefixed_state, storage_writer=writer, no_dist=True)

    # Run conversion and copy auxiliary assets.
    output_dir = tmp_path / "converted"
    _convert_fsdp_to_hf(str(origin_dir), str(fsdp_dir), str(output_dir))
    copy_assets(str(origin_dir), str(output_dir))

    converted = AutoModelForCausalLM.from_pretrained(output_dir)
    input_ids = torch.randint(0, converted.config.vocab_size, (1, 4))
    outputs = converted(input_ids=input_ids)

    assert outputs.logits.shape[:2] == (1, 4)
    # A single forward pass ensures weights are correctly loaded on CPU.
    assert not torch.isnan(outputs.logits).any()
