from diffusers import StableDiffusion3Pipeline
import torch
from torch import nn
from diffusers.quantizers import PipelineQuantizationConfig


class SD3(nn.Module):
    def __init__(self, args):
        super(SD3, self).__init__()
        self.device = args.device
        quantization_config = PipelineQuantizationConfig(
                                    quant_backend="bitsandbytes_8bit",
                                    quant_kwargs={"load_in_8bit": True, "bnb_8bit_quant_type": "nf4", "bnb_8bit_compute_dtype": torch.bfloat16},
                                    components_to_quantize=["transformer", "text_encoder_2"],
                                )
        self.pipeline = StableDiffusion3Pipeline.from_pretrained("stabilityai/stable-diffusion-3-medium-diffusers",
                                                                 torch_dtype=torch.float16,
                                                                 quantization_config=quantization_config)
        self.pipeline = self.pipeline.to(device=args.device)
        self.args = args

    def forward(self, prompt):
        print(prompt)
        image = self.pipeline(prompt,
                              generator=torch.Generator(device=self.args.device).manual_seed(0)).images[0]
        return image
