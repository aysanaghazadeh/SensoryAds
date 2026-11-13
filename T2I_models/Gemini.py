from torch import nn
from google import genai
import base64
from google.genai import types
import os
from PIL import Image
import io


class Gemini(nn.Module):
    def __init__(self, args):
        super(Gemini, self).__init__()
        api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise EnvironmentError("Set GOOGLE_API_KEY (or GEMINI_API_KEY) to use the Gemini T2I model.")
        self.client = genai.Client(api_key=api_key)
        self.model_name = getattr(args, "gemini_model", "gemini-3.1-flash-image")

    def forward(self, prompt, seed=None):
        try:
            if hasattr(self.client, "interactions"):
                interaction = self.client.interactions.create(
                    model=self.model_name,
                    input=prompt,
                )
                image = Image.open(io.BytesIO(base64.b64decode(interaction.output_image.data)))
            else:
                # Older google-genai without the interactions API
                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=prompt,
                    config=types.GenerateContentConfig(response_modalities=["IMAGE"]),
                )
                parts = response.candidates[0].content.parts
                image = next((Image.open(io.BytesIO(p.inline_data.data)) for p in parts if p.inline_data is not None), None)
                if image is None:
                    raise ValueError(f"No image in response: {response.text}")
        except Exception as e:
            print(f"Error generating image: {e}")
            image = None
        return image
