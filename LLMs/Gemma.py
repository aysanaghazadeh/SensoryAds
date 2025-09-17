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

    def forward(self, prompt):
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt}
                ]
            }
        ]

        output = self.pipe(text=messages)
        output = output[0]["generated_text"][-1]["content"]
        print(f'User: {prompt}')
        print(f'Assistant: {output}')
        return output