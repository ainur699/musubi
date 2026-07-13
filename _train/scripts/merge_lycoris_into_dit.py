#!/usr/bin/env python3
"""Offline-merge a LyCORIS (LoKr/LoHa/LoRA) checkpoint into the Qwen-Image DiT.

Produces a new base DiT safetensors with the adapter baked in, so a *fresh*
adapter can be trained on top and the two adapters kept separate (stacked
sequentially at inference: base + pretrain-adapter + new-adapter).

Musubi's training `--base_weights` path only supports musubi-native LoRA
(`create_arch_network_from_weights`), NOT LyCORIS, so we merge offline here,
mirroring the LyCORIS merge recipe used by the inference scripts
(`wan_generate_video.merge_lora_weights`, lycoris branch).

Example:
    conda run -n musubi python _train/scripts/merge_lycoris_into_dit.py \
      --dit _train/models/qwen_image_2512_bf16.safetensors \
      --lora_weight /home/a.gainetdinov/Github/eva-diffusion/output/qwen_image_live_t2i_lokr_f8/checkpoints/step=00010500/custom/lora_weights.safetensors \
      --lora_multiplier 1.0 \
      --save_merged_model _train/models/qwen_image_2512_bf16_merged_lork10500.safetensors \
      --device cpu
"""
import argparse
import logging

import torch
from safetensors.torch import load_file
from lycoris.kohya import create_network_from_weights

from musubi_tuner.qwen_image import qwen_image_model
from musubi_tuner.utils.safetensors_utils import mem_eff_save_file

logger = logging.getLogger("merge_lycoris_into_dit")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Merge a LyCORIS checkpoint into the Qwen-Image DiT.")
    p.add_argument("--dit", required=True, help="Base Qwen-Image DiT safetensors.")
    p.add_argument("--lora_weight", required=True, help="LyCORIS adapter safetensors to merge in.")
    p.add_argument("--lora_multiplier", type=float, default=1.0, help="Merge multiplier (default 1.0).")
    p.add_argument("--save_merged_model", required=True, help="Output path for the merged DiT.")
    p.add_argument("--device", default="cpu", help="Device for the merge (default cpu; GPUs may be busy).")
    p.add_argument("--attn_mode", default="torch", help="Attention mode for model creation (irrelevant to merge).")
    p.add_argument("--num_layers", type=int, default=60, help="Number of transformer blocks (Qwen-Image = 60).")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)

    logger.info(f"Loading base DiT from {args.dit} (device={device}, dtype=bf16)")
    model = qwen_image_model.load_qwen_image_model(
        device,            # device (merge/optim)
        args.dit,          # dit_path
        args.attn_mode,    # attn_mode
        False,             # split_attn
        False,             # zero_cond_t (not edit-2511)
        False,             # use_additional_t_cond (not layered)
        False,             # use_layer3d_rope (not layered)
        device,            # loading_device
        torch.bfloat16,    # dit_weight_dtype (LyCORIS merge requires bf16/fp16)
        fp8_scaled=False,
        lora_weights_list=None,
        lora_multipliers=None,
        num_layers=args.num_layers,
    )
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    n_keys_before = len(model.state_dict())
    logger.info(f"Base DiT loaded: {n_keys_before} tensors, {n_params/1e9:.2f}B params")

    logger.info(f"Loading LyCORIS adapter from {args.lora_weight}")
    weights_sd = load_file(args.lora_weight)
    logger.info(f"Adapter tensors: {len(weights_sd)}")

    lycoris_net, _ = create_network_from_weights(
        multiplier=args.lora_multiplier,
        file=None,
        weights_sd=weights_sd,
        unet=model,
        text_encoder=None,
        vae=None,
        for_inference=True,
    )
    logger.info(f"Merging adapter into DiT (multiplier={args.lora_multiplier})")
    with torch.no_grad():
        lycoris_net.merge_to(None, model, weights_sd, dtype=None, device=device)

    sd = model.state_dict()
    assert len(sd) == n_keys_before, f"key count changed after merge: {n_keys_before} -> {len(sd)}"
    logger.info(f"Saving merged model ({len(sd)} tensors) to {args.save_merged_model}")
    mem_eff_save_file(sd, args.save_merged_model)
    logger.info("Done.")


if __name__ == "__main__":
    main()
