from torch import nn
from MLLMs.InternVL2 import InternVL
from MLLMs.LLAVA16 import LLAVA16
from MLLMs.QWenVL import QWenVL
from MLLMs.GPT4_o import GPT4_o
from MLLMs.InternVL2 import InternVL

class MLLM(nn.Module):
    def __init__(self, args):
        super(MLLM, self).__init__()
        model_map = {
            'QWenVL': QWenVL,
            'LLAVA16': LLAVA16,
            'InternVL': InternVL,
            'GPT4_o': GPT4_o,
            'InternVL2_5': InternVL
        }
        self.model = model_map[args.MLLM](args)

    def forward(self, image, prompt, generate_kwargs=None):
        if generate_kwargs is None:
            output = self.model(image, prompt)
        else:
            output = self.model(image, prompt, generate_kwargs)
        return output

