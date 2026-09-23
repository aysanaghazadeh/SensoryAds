import json
from collections import defaultdict

from jinja2 import Environment, FileSystemLoader

from LLMs.LLM import LLM
from SensoryVisualElements.parsing import strip_code_fence


class CanonicalizationAgent:
    """Agent 2: walks the running object dictionary for a sensation and resolves each
    new raw object mention against it - folding it into an existing canonical entry
    when it refers to the same object (e.g. "ice" / "ice-cubes" / "ice cube"), or
    starting a new entry when it does not.

    Raw mentions are worked in batches of already-deduplicated (exact-string) counts
    rather than one at a time: each call sees the *entire* current dictionary plus a
    batch of new mentions, so a match is not dependent on which mention happened to
    arrive first, and the number of LLM calls stays proportional to the number of
    unique raw strings rather than the number of images.
    """

    def __init__(self, args):
        self.model = LLM(args)
        env = Environment(loader=FileSystemLoader(args.prompt_path))
        self.template = env.get_template(args.canonicalization_prompt)

    def canonicalize(self, sensation, raw_counts, batch_size):
        """raw_counts: dict[unique normalized raw string] -> occurrence count
        (already deduplicated by exact string match). Returns
        dict[canonical name] -> total occurrence count."""
        counts = defaultdict(int)
        raw_items = sorted(raw_counts.items(), key=lambda kv: -kv[1])
        for start in range(0, len(raw_items), batch_size):
            batch_items = raw_items[start:start + batch_size]
            batch = [raw for raw, _ in batch_items]
            resolved = self._resolve_batch(sensation, sorted(counts.keys()), batch)
            for raw, occurrence in batch_items:
                counts[resolved.get(raw, raw)] += occurrence
        return dict(counts)

    def _resolve_batch(self, sensation, canonical_names, batch):
        prompt = self.template.render(sensation=sensation, canonical_names=canonical_names, batch=batch)
        response = self.model(prompt)
        return self._parse(response, batch)

    @staticmethod
    def _parse(response, batch):
        text = strip_code_fence(response)
        try:
            mapping = json.loads(text)
        except json.JSONDecodeError:
            mapping = {}
        resolved = {}
        for raw in batch:
            canonical = mapping.get(raw) or raw
            resolved[raw] = ' '.join(str(canonical).strip().lower().split())
        return resolved
