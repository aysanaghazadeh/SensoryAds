from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch
from peft import PeftModel
import os


class Vicuna(nn.Module):
    def __init__(self, args):
        super(Vicuna, self).__init__()
        self.args = args
        if not args.train:
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True,
                bnb_8bit_compute_dtype=torch.float16
            )
            self.tokenizer = AutoTokenizer.from_pretrained("lmsys/vicuna-13b-v1.5",
                                                           token=os.environ.get('HF_TOKEN'))
            self.model = AutoModelForCausalLM.from_pretrained("lmsys/vicuna-13b-v1.5",
                                                              token=os.environ.get('HF_TOKEN'),
                                                              quantization_config=quantization_config,
                                                              device_map="auto")
            if args.fine_tuned:
                self.model = PeftModel.from_pretrained(self.model, os.path.join(args.model_path, 'my_ppo_model_DMD_batch_size_1'))

    def forward(self, inputs):
        if not self.args.train:
            inputs = self.tokenizer(inputs, return_tensors="pt").to(device=self.args.device)
            generated_ids = self.model.generate(**inputs, max_new_tokens=512, temperature=0)
            output = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
            output = output.replace('</s>', '')
            output = output.replace("['", '')
            output = output.replace("']", '')
            output = output.replace('["', '')
            output = output.replace('"]', '')
            output = output.split(':')[-1]
            return output
        # return self.model(**inputs)
