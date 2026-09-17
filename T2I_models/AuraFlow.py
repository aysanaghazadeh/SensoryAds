import torch
from torch import nn
from diffusers import AuraFlowPipeline
from diffusers.quantizers import PipelineQuantizationConfig


class AuraFlow(nn.Module):
    def __init__(self, args):
        super(AuraFlow, self).__init__()
        self.device = args.device
        quantization_config = PipelineQuantizationConfig(
            quant_backend="bitsandbytes_8bit",
            quant_kwargs={"load_in_8bit": True, "bnb_8bit_quant_type": "nf4", "bnb_8bit_compute_dtype": torch.bfloat16},
            components_to_quantize=["transformer", "text_encoder_2"],
        )
        self.pipeline = AuraFlowPipeline.from_pretrained(
            "fal/AuraFlow-v0.3",
            torch_dtype=torch.float16,
            device_map='balanced',
            # variant="fp16",
            quantization_config=quantization_config
        )

    def forward(self, prompt, seed=None):
        seed = seed if seed is not None else 0
        image = self.pipeline(
                    prompt=prompt,
                    height=1024,
                    width=1024,
                    num_inference_steps=28,
                    generator=torch.Generator().manual_seed(seed),
                    guidance_scale=5,
                    ).images[0]
        return image
