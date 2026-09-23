import json

from PIL import Image
from jinja2 import Environment, FileSystemLoader

from MLLMs.MLLM import MLLM
from SensoryVisualElements.parsing import strip_code_fence

# These backends' forward(image, prompt, ...) implementations take a raw file
# path instead of a PIL Image (GPT4_o does its own open()+base64 encoding), and/or
# don't accept a generate_kwargs argument at all (GPT4_o, Gemini). Every other
# backend (InternVL, QWenVL, LLAVA16, Gemma, MOLMO, FastVLM...) follows the
# convention used elsewhere in this repo (see generation/description_generation.py):
# open the image with PIL first and pass generate_kwargs through.
PATH_INPUT_BACKENDS = {'GPT4_o'}
NO_GENERATE_KWARGS_BACKENDS = {'GPT4_o', 'Gemini'}


class ObjectExtractionAgent:
    """Agent 1: looks at a single image and lists the concrete visual objects it sees.

    Deliberately does not try to de-duplicate across images itself - that is the
    canonicalization agent's job. It only normalizes its own output (lowercase,
    singular-ish phrasing) so the second agent has less noise to resolve.
    """

    def __init__(self, args):
        self.model = MLLM(args)
        self.mllm_name = args.MLLM
        self.max_new_tokens = args.extraction_max_new_tokens
        env = Environment(loader=FileSystemLoader(args.prompt_path))
        self.template = env.get_template(args.extraction_prompt)

    def extract(self, image_path, sensation):
        prompt = self.template.render(sensation=sensation)
        image_input = image_path if self.mllm_name in PATH_INPUT_BACKENDS else Image.open(image_path)
        if self.mllm_name in NO_GENERATE_KWARGS_BACKENDS:
            response = self.model(image_input, prompt)
        else:
            response = self.model(image_input, prompt, generate_kwargs={"max_new_tokens": self.max_new_tokens})
        return self._parse(response)

    @staticmethod
    def _parse(response):
        text = strip_code_fence(response)
        try:
            objects = json.loads(text)
        except json.JSONDecodeError:
            objects = [item.strip(' -*"\'') for item in text.replace('\n', ',').split(',')]
        if isinstance(objects, dict):
            objects = objects.get('objects', [])
        return [' '.join(str(item).strip().lower().split()) for item in objects if str(item).strip()]
