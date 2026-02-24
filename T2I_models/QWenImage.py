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
        # self.pipe = DiffusionPipeline.from_pretrained(model_name,
        #                                               torch_dtype=torch_dtype,
        #                                               quantization_config=quantization_config)
        # self.args = args
        # self.pipe = self.pipe.to(device=args.device)
        # From https://github.com/ModelTC/Qwen-Image-Lightning/blob/342260e8f5468d2f24d084ce04f55e101007118b/generate_with_diffusers.py#L82C9-L97C10
        scheduler_config = {
            "base_image_seq_len": 256,
            "base_shift": math.log(3),  # We use shift=3 in distillation
            "invert_sigmas": False,
            "max_image_seq_len": 8192,
            "max_shift": math.log(3),  # We use shift=3 in distillation
            "num_train_timesteps": 1000,
            "shift": 1.0,
            "shift_terminal": None,  # set shift_terminal to None
            "stochastic_sampling": False,
            "time_shift_type": "exponential",
            "use_beta_sigmas": False,
            "use_dynamic_shifting": True,
            "use_exponential_sigmas": False,
            "use_karras_sigmas": False,
        }
        scheduler = FlowMatchEulerDiscreteScheduler.from_config(scheduler_config)
        self.pipe = DiffusionPipeline.from_pretrained(
                "Qwen/Qwen-Image", 
                scheduler=scheduler,
                torch_dtype=torch.bfloat16, 
                quantization_config=quantization_config,
                device_map='balanced'
            )
        self.pipe.load_lora_weights(
            "lightx2v/Qwen-Image-Lightning", weight_name="Qwen-Image-Lightning-8steps-V1.0.safetensors"
        )
        wandb.init(project="QWenImage")

    def forward(self, prompt, seed=None):
        seed = seed if seed is not None else 0
        # positive_magic = {
        #     "en": ", Ultra HD, 4K, cinematic composition.",  # for english prompt
        #     "zh": ", 超清，4K，电影级构图."  # for chinese prompt
        # }

        # negative_prompt = " "  # using an empty string if you do not have specific concept to remove

        # # Generate with different aspect ratios
        # aspect_ratios = {
        #     "1:1": (1328, 1328),
        #     "16:9": (1664, 928),
        #     "9:16": (928, 1664),
        #     "4:3": (1472, 1140),
        #     "3:4": (1140, 1472),
        #     "3:2": (1584, 1056),
        #     "2:3": (1056, 1584),
        # }

        # width, height = aspect_ratios["16:9"]

        # image = self.pipe(
        #     prompt=prompt + positive_magic["en"],
        #     negative_prompt=negative_prompt,
        #     width=width,
        #     height=height,
        #     num_inference_steps=28,
        #     true_cfg_scale=4.0,
        #     generator=torch.Generator(device="cuda").manual_seed(seed)
        # ).images[0]
        image = self.pipe(
            prompt=prompt,
            width=1024,
            height=1024,
            num_inference_steps=8,
            generator=torch.manual_seed(seed),
        ).images[0]
        wandb.log({
            "image": wandb.Image(image, caption=prompt),
            "prompt": prompt,
        })
        return image