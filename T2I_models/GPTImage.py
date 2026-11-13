from torch import nn
from openai import OpenAI
import base64
from PIL import Image
import io


class GPTImage(nn.Module):
    def __init__(self, args):
        super(GPTImage, self).__init__()
        self.client = OpenAI()
        self.model_name = getattr(args, "openai_image_model", "gpt-image-2.5-sunburst")

    def forward(self, prompt, seed=None):
        try:
            result = self.client.images.generate(model=self.model_name, prompt=prompt)
            image = Image.open(io.BytesIO(base64.b64decode(result.data[0].b64_json)))
        except Exception as e:
            print(f"Error generating image: {e}")
            image = None
        return image
