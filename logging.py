import os
import re
import glob
import random
from typing import List, Dict, Optional

import torch
import wandb
from PIL import Image

from diffusers import FluxControlPipeline
from diffusers.utils import load_image


# ----------------------------
# Helpers
# ----------------------------

def _natural_key(s: str):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s)]

def find_lora_checkpoints(ckpt_root: str, pattern: str = "checkpoint-*") -> List[str]:
    paths = glob.glob(os.path.join(ckpt_root, pattern))
    paths = [p for p in paths if os.path.isdir(p)]
    return sorted(paths, key=_natural_key)

def seed_everything(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def parse_step_from_ckpt_dir(ckpt_dir: str) -> Optional[int]:
    m = re.search(r"checkpoint-(\d+)", os.path.basename(ckpt_dir))
    return int(m.group(1)) if m else None

def load_lora_into_pipe(
    pipe,
    lora_dir: str,
    adapter_name: str = "lora",
    weight_name: Optional[str] = None,
    scale: float = 1.0,
):
    # prevent LoRA stacking
    if hasattr(pipe, "unload_lora_weights"):
        try:
            pipe.unload_lora_weights()
        except Exception:
            pass

    kwargs = {"adapter_name": adapter_name}
    if weight_name is not None:
        kwargs["weight_name"] = weight_name

    pipe.load_lora_weights(lora_dir, **kwargs)

    # set LoRA scale (newer diffusers)
    if hasattr(pipe, "set_adapters"):
        pipe.set_adapters([adapter_name], adapter_weights=[scale])
    elif hasattr(pipe, "fuse_lora"):
        try:
            pipe.fuse_lora(lora_scale=scale)
        except Exception:
            pass


def generate_images_for_prompts_with_control(
    pipe,
    prompts: List[str],
    *,
    control_image: Image.Image,
    control_arg_name: str = "control_image",  # <-- change if needed
    negative_prompt: Optional[str] = None,
    num_inference_steps: int = 28,
    guidance_scale: float = 3.5,
    height: int = 1024,
    width: int = 1024,
    seed: int = 0,
    device: str = "cuda",
    extra_call_kwargs: Optional[Dict] = None,
) -> List[Image.Image]:
    extra_call_kwargs = extra_call_kwargs or {}

    seed_everything(seed)
    generator = torch.Generator(device=device).manual_seed(seed)

    pipe.to(device)

    images: List[Image.Image] = []
    call_kwargs = dict(
        negative_prompt=negative_prompt,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        height=height,
        width=width,
        generator=generator,
        **extra_call_kwargs,
    )
    call_kwargs[control_arg_name] = control_image

    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        for p in prompts:
            out = pipe(prompt=p, **call_kwargs)
            images.append(out.images[0])

    return images


# ----------------------------
# Main: evaluate all checkpoints + log to W&B
# ----------------------------

def log_checkpoints_to_wandb_with_control(
    *,
    ckpt_root: str,
    run_name: str,
    project: str,
    prompts: List[str],
    control_image_path: str,
    base_model_id_or_path: str,
    control_arg_name: str = "control_image",  # <-- change if your pipeline uses another key
    negative_prompt: Optional[str] = None,
    num_inference_steps: int = 28,
    guidance_scale: float = 3.5,
    height: int = 1024,
    width: int = 1024,
    seed: int = 0,
    lora_scale: float = 1.0,
    lora_weight_name: Optional[str] = None,
    extra_call_kwargs: Optional[Dict] = None,
    every_n: int = 1,
):
    ckpt_dirs = ['checkpoint-100', 'checkpoint-200', 'checkpoint-300', 'checkpoint-400', 'checkpoint-500', 'checkpoint-600', 'checkpoint-700', 'checkpoint-800', 'checkpoint-900', 'checkpoint-1100', 'checkpoint-1200']
    if not ckpt_dirs:
        raise FileNotFoundError(f"No checkpoints found under: {ckpt_root}")

    # Load control image ONCE
    control_img = load_image(control_image_path).convert("RGB")

    wandb.init(
        project=project,
        name=run_name,
        config={
            "ckpt_root": ckpt_root,
            "base_model": base_model_id_or_path,
            "control_image_path": control_image_path,
            "control_arg_name": control_arg_name,
            "num_inference_steps": num_inference_steps,
            "guidance_scale": guidance_scale,
            "height": height,
            "width": width,
            "seed": seed,
            "lora_scale": lora_scale,
            "prompts": prompts,
            "negative_prompt": negative_prompt,
            "every_n": every_n,
        },
    )

    pipe = FluxControlPipeline.from_pretrained(
        base_model_id_or_path,
        torch_dtype=torch.bfloat16,
    )

    if hasattr(pipe, "enable_attention_slicing"):
        pipe.enable_attention_slicing()
    if hasattr(pipe, "enable_vae_slicing"):
        pipe.enable_vae_slicing()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    for i, ckpt_dir in enumerate(ckpt_dirs):
        if (i % every_n) != 0:
            continue

        step = parse_step_from_ckpt_dir(ckpt_dir)
        step_for_wandb = step if step is not None else i

        load_lora_into_pipe(
            pipe,
            lora_dir=ckpt_dir,
            adapter_name="lora",
            weight_name=lora_weight_name,
            scale=lora_scale,
        )

        imgs = generate_images_for_prompts_with_control(
            pipe,
            prompts,
            control_image=control_img,
            control_arg_name=control_arg_name,
            negative_prompt=negative_prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            height=height,
            width=width,
            seed=seed,
            device=device,
            extra_call_kwargs=extra_call_kwargs,
        )

        wandb.log(
            {
                "ckpt_dir": os.path.basename(ckpt_dir),
                "control_image": wandb.Image(control_img, caption="control"),
                "samples": [wandb.Image(im, caption=f"step={step_for_wandb} | {p}") for im, p in zip(imgs, prompts)],
            },
            step=step_for_wandb,
        )

    wandb.finish()


# ----------------------------
# Example usage
# ----------------------------
if __name__ == "__main__":
    log_checkpoints_to_wandb_with_control(
        ckpt_root="../models/flux_edit_lora",
        run_name="fluxcontrol-lora-eval-with-control",
        project="fluxcontrol",
        prompts=[
            "Generate an image that evokes Comforting Warmth and conveys I should buy this car because it is luxurious.",
        ],
        control_image_path="../Data/PittAd/train_images_total/4/13534.jpg",
        base_model_id_or_path="black-forest-labs/FLUX.1-dev",
        control_arg_name="control_image",   # change if your pipeline expects different kwarg
        num_inference_steps=28,
        guidance_scale=3.5,
        height=1024,
        width=1024,
        seed=0,
        lora_scale=1.0,
        lora_weight_name=None,
        extra_call_kwargs=None,
        every_n=1,
    )
