from diffusers import FluxPipeline, DiffusionPipeline
from diffusers import FlowMatchEulerDiscreteScheduler
import torch
from torch import nn
from diffusers.quantizers import PipelineQuantizationConfig
import wandb
import math


class QWenImage(nn.Module):
    def __init__(self, args):
        super(QWenImage, self).__init__()
        self.device = args.device
        model_name = "Qwen/Qwen-Image"

        # Load the pipeline
        if torch.cuda.is_available():
            torch_dtype = torch.bfloat16
        quantization_config = PipelineQuantizationConfig(
            quant_backend="bitsandbytes_4bit",
            quant_kwargs={"load_in_4bit": True, "bnb_4bit_quant_type": "nf4", "bnb_4bit_compute_dtype": torch.bfloat16},
            components_to_quantize=["text_encoder"],
        )
        self.pipe = DiffusionPipeline.from_pretrained(
                "Qwen/Qwen-Image", 
                torch_dtype=torch.bfloat16, 
                quantization_config=quantization_config,
                device_map='balanced'
            )
        wandb.init(project="QWenImage")

    def forward(self, prompt, seed=None):
        seed = seed if seed is not None else 0
        positive_magic = {
            "en": ", Ultra HD, 4K, cinematic composition.",  # for english prompt
            "zh": ", 超清，4K，电影级构图."  # for chinese prompt
        }

        negative_prompt = " "  # using an empty string if you do not have specific concept to remove
        image = self.pipe(
            prompt=prompt + positive_magic["en"],
            negative_prompt=negative_prompt,
            width=1024,
            height=1024,
            num_inference_steps=28,
            true_cfg_scale=4.0,
            generator=torch.manual_seed(seed),
        ).images[0]
        wandb.log({
            "image": wandb.Image(image, caption=prompt),
            "prompt": prompt,
        })
        return image