from T2I_models.T2I_model import T2IModel
from utils.prompt_engineering.prompt_generation import ImageGenerationPromptGenerator
from torch import nn


class AdvertisementImageGeneration(nn.Module):
    def __init__(self, args):
        super(AdvertisementImageGeneration, self).__init__()
        self.args = args
        self.prompt_generator = ImageGenerationPromptGenerator(self.args)
        self.T2I_model = T2IModel(args)
        if args.text_input_type == 'LLM':
            self.prompt_generator.set_LLM(args)

    def forward(self, image_filename, sensation=None, prompt=None, seed=None, generated_image=None):
        if prompt is None:
            prompt = self.prompt_generator.generate_prompt(self.args, image_filename, sensation)
        if generated_image is not None:
            image = self.T2I_model(image_filename, prompt, seed, generated_image, sensation)
        else:
            image = self.T2I_model(image_filename, prompt, seed)
        return image, prompt
