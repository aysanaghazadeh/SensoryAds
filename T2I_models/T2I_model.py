from torch import nn
from T2I_models.PitxArt import PixArt
from T2I_models.SDXLFlash import SDXL
from T2I_models.DMD2 import DMD
from T2I_models.AuraFlow import AuraFlow
from T2I_models.DALLE3 import DALLE3
from T2I_models.Flux import Flux
from T2I_models.QWenImage import QWenImage
from T2I_models.SD3 import SD3

class T2IModel(nn.Module):
    def __init__(self, args):
        super(T2IModel, self).__init__()
        model_map = {
            'PixArt': PixArt,
            'SDXL': SDXL,
            'DMD': DMD,
            'AuraFlow': AuraFlow,
            'DALLE3': DALLE3,
            'Flux': Flux,
            'QWenImage': QWenImage,
            'SD3': SD3
        }
        self.model = model_map[args.T2I_model](args)

    def forward(self, prompt, seed=None):
        return self.model(prompt, seed)
