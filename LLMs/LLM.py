from torch import nn
from LLMs.Mistral7B import Mistral7B
from LLMs.LLAMA3 import LLAMA3
from LLMs.phi import Phi
from LLMs.Mistral7BInstruct import Mistral7BInstruct
from LLMs.vicuna import Vicuna
from LLMs.LLAMA3_instruct import LLAMA3Instruct
from LLMs.InternLM import InternLM
from LLMs.QWenLM import QWenLM
from LLMs.GPT4o import GPT4o
from LLMs.Gemma import Gemma


class LLM(nn.Module):
    def __init__(self, args):
        super(LLM, self).__init__()
        model_map = {
            'Mistral7B': Mistral7B,
            'LLAMA3': LLAMA3,
            'LLAMA3_instruct': LLAMA3Instruct,
            'phi': Phi,
            'Mistral7BInstruct': Mistral7BInstruct,
            'vicuna': Vicuna,
            'InternLM': InternLM,
            'QWenLM': QWenLM,
            'GPT4o': GPT4o,
            'Gemma': Gemma,
        }
        self.model = model_map[args.LLM](args)

    def forward(self, prompt):
        output = self.model(prompt)
        return output

