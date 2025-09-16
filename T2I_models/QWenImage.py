from diffusers import FluxPipeline, DiffusionPipeline
import torch
from torch import nn
from diffusers.quantizers import PipelineQuantizationConfig


class QWenImage(nn.Module):
    def __init__(self, args):
        super(QWenImage, self).__init__()
        self.device = args.device
        quantization_config = PipelineQuantizationConfig(
            quant_backend="bitsandbytes_4bit",
            quant_kwargs={"load_in_4bit": True, "bnb_8bit_quant_type": "nf4", "bnb_4bit_compute_dtype": torch.bfloat16},
            components_to_quantize=["transformer", "text_encoder_2"],
        )
        self.pipe = DiffusionPipeline.from_pretrained("Qwen/Qwen-Image",
                                                           torch_dtype=torch.float16,
                                                          quantization_config=quantization_config)
        self.args = args
        self.pipe = self.pipe.to(device=args.device)

    def forward(self, prompt):
        positive_magic = {
            "en": ", Ultra HD, 4K, cinematic composition.",  # for english prompt
            "zh": ", 超清，4K，电影级构图."  # for chinese prompt
        }
        negative_prompt = " "  # using an empty string if you do not have specific concept to remove

        # Generate with different aspect ratios
        aspect_ratios = {
            "1:1": (1328, 1328),
            "16:9": (1664, 928),
            "9:16": (928, 1664),
            "4:3": (1472, 1140),
            "3:4": (1140, 1472),
            "3:2": (1584, 1056),
            "2:3": (1056, 1584),
        }
        width, height = aspect_ratios["16:9"]

        image = self.pipe(
            prompt=prompt + positive_magic["en"],
            negative_prompt=negative_prompt,
            width=width,
            height=height,
            num_inference_steps=28,
            true_cfg_scale=4.0,
            generator=torch.Generator(device="cuda").manual_seed(0)
        ).images[0]

        return image
