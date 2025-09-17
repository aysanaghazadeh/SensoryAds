from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch
from peft import PeftModel
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


class InternLM(nn.Module):
    def __init__(self, args):
        super(InternLM, self).__init__()
        self.args = args
        if not args.train:
            self.args = args
            self.tokenizer = AutoTokenizer.from_pretrained("internlm/internlm3-8b-instruct", trust_remote_code=True)
            self.model = AutoModelForCausalLM.from_pretrained("internlm/internlm3-8b-instruct",
                                                              torch_dtype=torch.float16,
                                                              load_in_8bit=True,
                                                              trust_remote_code=True)
            self.model = self.model.eval()

    def forward(self, prompt):
        if not self.args.train:
            messages = [
                {"role": "user", "content": prompt},
            ]
            tokenized_chat = self.tokenizer.apply_chat_template(messages,
                                                                tokenize=True,
                                                                add_generation_prompt=True,
                                                                return_tensors="pt").to(self.args.device)

            generated_ids = self.model.generate(tokenized_chat)

            generated_ids = [
                output_ids[len(input_ids):] for input_ids, output_ids in zip(tokenized_chat, generated_ids)
            ]
            # prompt = self.tokenizer.batch_decode(tokenized_chat)[0]
            print(f'User: {prompt}')
            response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
            print(f'Assistant: {response}')
            return response