from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig, BitsAndBytesConfig
from torch import nn


class MOLMO(nn.Module):
    def __init__(self, args):
        super().__init__()

        bnb_config = BitsAndBytesConfig(load_in_8bit=True)
        self.model = AutoModelForCausalLM.from_pretrained('allenai/Molmo-7B-O-0924',
                                                          trust_remote_code=True,
                                                          torch_dtype='auto',
                                                          device_map='auto',
                                                          quantization_config=bnb_config
                                                          )
        self.processor = AutoProcessor.from_pretrained('allenai/Molmo-7B-O-0924',
                                                       trust_remote_code=True,
                                                       torch_dtype='auto',
                                                       device_map='auto'
                                                       )

    def forward(self, image, prompt, generate_kwargs={"max_new_tokens": 250}):
        inputs = self.processor.process(
            images=[image],
            text=prompt
        )
        inputs = {k: v.to(self.model.device).unsqueeze(0) for k, v in inputs.items()}

        # generate output; maximum 200 new tokens; stop generation when <|endoftext|> is generated
        output = self.model.generate_from_batch(
            inputs,
            GenerationConfig(max_new_tokens=generate_kwargs["max_new_tokens"], stop_strings="<|endoftext|>"),
            tokenizer= self.processor.tokenizer
        )

        # only get generated tokens; decode them to text
        generated_tokens = output[0, inputs['input_ids'].size(1):]
        output = self.processor.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        print(f'User: {prompt}')
        print(f'Assistant: {output}')
        return output