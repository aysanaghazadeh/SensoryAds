from torch import nn
from transformers import BitsAndBytesConfig, pipeline, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch
from peft import PeftModel
import os


class LLAMA3Instruct(nn.Module):
    def __init__(self, args):
        super(LLAMA3Instruct, self).__init__()
        self.args = args
        if not args.train:
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True,
                bnb_8bit_compute_dtype=torch.float16
            )
            if args.fine_tuned:
                self.model = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct",
                                                             token=os.environ.get('HF_TOKEN'),
                                                             device_map='auto')
                self.tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct",
                                                          token=os.environ.get('HF_TOKEN'))
                self.tokenizer.pad_token = self.tokenizer.eos_token
                self.tokenizer.padding_side = "right"
                self.model = PeftModel.from_pretrained(self.model,
                                                       os.path.join(args.model_path,
                                                                    'my_HierarchicalCPO_LLAMA3_instruct/checkpoint-1000'))
            else:
                model_id = "meta-llama/Meta-Llama-3-8B-Instruct"

                # self.pipeline = pipeline(
                #     "text-generation",
                #     model=model_id,
                #     token=os.environ.get('HF_TOKEN'),
                #     model_kwargs={"torch_dtype": torch.bfloat16},
                #     quantization_config=BitsAndBytesConfig(load_in_8bit=True),
                #     device_map="auto",
                # )
                self.model = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct",
                                                                  token=os.environ.get('HF_TOKEN'),
                                                                  load_in_8bit=True,
                                                                  device_map='auto')
                self.tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct",
                                                               token=os.environ.get('HF_TOKEN'))
        else:
            self.model = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct",
                                                              token=os.environ.get('HF_TOKEN'),
                                                              device_map='auto')
            self.tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct",
                                                           token=os.environ.get('HF_TOKEN'),
                                                           trust_remote_code=True,
                                                           padding='right')

    def forward(self, prompt):
        if not self.args.fine_tuned:
            # print('llm prompt:', prompt)
            messages = [
                {"role": "system", "content": "Be a helpful assistant"},
                {"role": "user", "content": prompt},
            ]
            input_ids = self.tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                return_tensors="pt"
            ).to(self.model.device)

            terminators = [
                self.tokenizer.eos_token_id,
                self.tokenizer.convert_tokens_to_ids("<|eot_id|>")
            ]

            outputs = self.model.generate(
                input_ids,
                max_new_tokens=256,
                eos_token_id=terminators,
                do_sample=True,
                temperature=0.6,
                top_p=0.9,
            )
            response = outputs[0][input_ids.shape[-1]:]
            response = self.tokenizer.decode(response, skip_special_tokens=True)
            print(f'User: {prompt}')
            print(f'Assistant: {response}')
            return response
        else:
            messages = [
                {"role": "system", "content": "Be a helpful assistant"},
                {"role": "user", "content": prompt},
            ]
            input_ids = self.tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                return_tensors="pt"
            ).to(self.model.device)

            terminators = [
                self.tokenizer.eos_token_id,
                self.tokenizer.convert_tokens_to_ids("<|eot_id|>")
            ]
            # input_ids = self.tokenizer(prompt, return_tensor=True)
            outputs = self.model.generate(
                input_ids,
                max_new_tokens=25,
                eos_token_id=terminators,
                do_sample=True,
                temperature=0.6,
                top_p=0.9,
            )
            response = outputs[0][input_ids.shape[-1]:]
            output = self.tokenizer.decode(response, skip_special_tokens=True)
            return output
        # return self.model(**inputs)
