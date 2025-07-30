from torch import nn
from transformers import pipeline


class Mixtral7B(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.pipe = pipeline("text-generation",
                             model="mistralai/Mistral-7B-v0.1",
                             device=args.device,
                             device_map=[0, 1, 2, 3],
                             max_length=150)

    def forward(self, prompt):
        return self.pipe(prompt)
