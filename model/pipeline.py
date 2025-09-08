from T2I_models.T2I_model import T2IModel
from utils.prompt_engineering.prompt_generation import PromptGenerator
from torch import nn


class AdvertisementImageGeneration(nn.Module):
    def __init__(self, args):
        super(AdvertisementImageGeneration, self).__init__()
        self.args = args
        self.prompt_generator = PromptGenerator(self.args)
        self.T2I_model = T2IModel(args)
        if args.text_input_type == 'LLM':
            self.prompt_generator.set_LLM(args)

    def forward(self, image_filename, sensation=None, prompt=None):
        if prompt is None:
            prompt = self.prompt_generator.generate_prompt(self.args, image_filename, sensation)
        image = self.T2I_model(prompt)
        return image, prompt
