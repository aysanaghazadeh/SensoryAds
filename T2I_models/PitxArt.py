import torch
from torch import nn
from diffusers.quantizers import PipelineQuantizationConfig
from diffusers import PixArtAlphaPipeline

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
                                                        torch_dtype=torch.float16,
                                                        quantization_config=quantization_config)
        self.pipe = self.pipe.to(device=args.device)

    def forward(self, prompt, seed=None):
        image = self.pipe(prompt,
                          generator=torch.Generator(device=self.args.device).manual_seed(seed)).images[0]
        return image
