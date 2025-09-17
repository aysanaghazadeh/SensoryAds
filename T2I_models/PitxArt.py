import torch
from torch import nn
from diffusers.quantizers import PipelineQuantizationConfig


class PixArt(nn.Module):
    def __init__(self, args):
        super(PixArt, self).__init__()
        self.device = args.device
        quantization_config = PipelineQuantizationConfig(
            quant_backend="bitsandbytes_8bit",
            quant_kwargs={"load_in_8bit": True, "bnb_8bit_quant_type": "nf4", "bnb_8bit_compute_dtype": torch.bfloat16},
            components_to_quantize=["transformer", "text_encoder_2"],
        )
        self.pipe = PixArtAlphaPipeline.from_pretrained("PixArt-alpha/PixArt-XL-2-1024-MS",
                                                        torch_dtype=torch.float16)
        self.pipe = self.pipe.to(device=args.device)

    def forward(self, prompt):
        image = self.pipe(prompt).images[0]
        return image
