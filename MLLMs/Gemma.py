from transformers import pipeline
import torch


class Gemma(torch.nn.Module):
    def __init__(self, args):
        super().__init__()

        self.pipe = pipeline(
                        "image-text-to-text",
                        model="google/gemma-3-4b-it",
                        device=args.device,
                        torch_dtype=torch.bfloat16
                    )

    def forward(self, image, prompt, generate_kwargs={"max_new_tokens": 250}):
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image",
                     "image": image},
                    {"type": "text", "text": prompt}
                ]
            }
        ]

        output = self.pipe(text=messages, max_new_tokens=generate_kwargs["max_new_tokens"])
        return output