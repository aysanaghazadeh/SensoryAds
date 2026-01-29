from torch import nn
from T2I_models.PitxArt import PixArt
from T2I_models.SDXLFlash import SDXL
from T2I_models.DMD2 import DMD
from T2I_models.AuraFlow import AuraFlow
from T2I_models.DALLE3 import DALLE3
from T2I_models.Flux import Flux
from T2I_models.QWenImage import QWenImage
from T2I_models.SD3 import SD3
from Editing.agents import ImageEditingAgent

class T2IModel(nn.Module):
    def __init__(self, args):
        super(T2IModel, self).__init__()
        self.args = args
        model_map = {
            'PixArt': PixArt,
            'SDXL': SDXL,
            'DMD': DMD,
            'AuraFlow': AuraFlow,
            'DALLE3': DALLE3,
            'Flux': Flux,
            'QWenImage': QWenImage,
            'SD3': SD3,
            'AgenticEditing': ImageEditingAgent
        }
        self.model = model_map[args.T2I_model](args)

    def forward(self, filename, prompt, seed=None, generated_image=None, target_sensation_initial=None):
        if self.args.T2I_model == 'AgenticEditing':
            return self.model.agentic_image_editing(filename, generated_image, prompt, target_sensation_initial)
        else:
            return self.model(prompt, seed)
