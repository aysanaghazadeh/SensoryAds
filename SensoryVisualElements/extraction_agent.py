import json

from jinja2 import Environment, FileSystemLoader

from MLLMs.MLLM import MLLM
from SensoryVisualElements.parsing import strip_code_fence


class ObjectExtractionAgent:
    """Agent 1: looks at a single image and lists the concrete visual objects it sees.

    Deliberately does not try to de-duplicate across images itself - that is the
    canonicalization agent's job. It only normalizes its own output (lowercase,
    singular-ish phrasing) so the second agent has less noise to resolve.
    """

    def __init__(self, args):
        self.model = MLLM(args)
        env = Environment(loader=FileSystemLoader(args.prompt_path))
        self.template = env.get_template(args.extraction_prompt)

    def extract(self, image_path, sensation):
        prompt = self.template.render(sensation=sensation)
        response = self.model(image_path, prompt)
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
