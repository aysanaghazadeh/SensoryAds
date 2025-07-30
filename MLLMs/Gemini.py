from google.genai import types
import base64
import requests
from torch import nn
from google import genai

class Gemini(nn.Module):
    def __init__(self, args):
        super(Gemini, self).__init__()
        self.client = genai.Client()
        

    def forward(self, image, prompt):
        response = self.client.models.generate_content(
                        model="gemini-2.5-flash",
                        contents=[image, prompt],
                    )
        print(f'Gemini Reponse: {response.text}')
        print('-' * 40)
        return response.text.split(':')[-1]
        