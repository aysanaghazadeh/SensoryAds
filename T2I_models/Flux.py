from diffusers import FluxPipeline, DiffusionPipeline
import torch
from torch import nn
from diffusers.quantizers import PipelineQuantizationConfig


class Flux(nn.Module):
    def __init__(self, args):
        super(Flux, self).__init__()
        self.device = args.device
        quantization_config = PipelineQuantizationConfig(
                                    quant_backend="bitsandbytes_8bit",
                                    quant_kwargs={"load_in_8bit": True, "bnb_8bit_quant_type": "nf4", "bnb_8bit_compute_dtype": torch.bfloat16},
                                    components_to_quantize=["transformer", "text_encoder_2"],
                                )
        self.pipeline = DiffusionPipeline.from_pretrained("black-forest-labs/FLUX.1-dev",
                                                           torch_dtype=torch.float16,
                                                          quantization_config=quantization_config)
        self.pipeline = self.pipeline.to(device=args.device)
        self.args = args

    def forward(self, prompt, seed=None):
        print(prompt)
        seed = seed if seed is not None else 0
        image = self.pipeline(prompt,
                              generator=torch.Generator(device=self.args.device).manual_seed(seed)).images[0]
        return image
