from torch import nn
from google import genai
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
        self.model_name = getattr(args, "gemini_model", "gemini-2.5-flash-image")

    def forward(self, prompt, seed=None):
        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=types.GenerateContentConfig(response_modalities=["IMAGE"]),
            )
            image = None
            for part in response.candidates[0].content.parts:
                if part.inline_data is not None:
                    image = Image.open(io.BytesIO(part.inline_data.data))
                    break
            if image is None:
                raise ValueError(f"No image in response: {response.text}")
        except Exception as e:
            print(f"Error generating image: {e}")
            image = None
        return image
