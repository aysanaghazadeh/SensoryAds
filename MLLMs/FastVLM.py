from transformers import BitsAndBytesConfig,AutoModelForCausalLM, AutoTokenizer
from qwen_vl_utils import process_vision_info
import torch


class FastVLM(torch.nn.Module):
    def __init__(self, args):
        super().__init__()
        bnb_config = BitsAndBytesConfig(load_in_8bit=True)
        MID='apple/FastVLM-7B'
        self.model = AutoModelForCausalLM.from_pretrained(MID,
                                                         torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                                                         device_map="auto",
                                                         quantization_config=bnb_config,
                                                         trust_remote_code=True,
                                                         )
        # self.model = self.model.to(device=args.device)
        self.model.eval()
        self.IMAGE_TOKEN_INDEX = -200
        # self.model = self.model.to(device=args.device)

        self.tokenizer = AutoTokenizer.from_pretrained(MID, trust_remote_code=True)


    def forward(self, image, prompt, generate_kwargs):
        if '<image>' not in prompt:
            prompt += '<image>\n'
        messages = [
            {"role": "user", "content": prompt}
        ]

        text = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )
        pre, post = text.split("<image>", 1)
        pre_ids = self.tokenizer(pre, return_tensors="pt", add_special_tokens=False).input_ids
        post_ids = self.tokenizer(post, return_tensors="pt", add_special_tokens=False).input_ids
        # Splice in the IMAGE token id (-200) at the placeholder position
        img_tok = torch.tensor([[self.IMAGE_TOKEN_INDEX]], dtype=pre_ids.dtype)
        input_ids = torch.cat([pre_ids, img_tok, post_ids], dim=1).to(self.model.device)
        attention_mask = torch.ones_like(input_ids, device=self.model.device)
        # Preprocess image via the model's own processor
        img = image
        px = self.model.get_vision_tower().image_processor(images=img, return_tensors="pt")["pixel_values"]
        px = px.to(self.model.device, dtype=self.model.dtype)
        # Generate
        with torch.no_grad():
            out = self.model.generate(
                inputs=input_ids,
                attention_mask=attention_mask,
                images=px,
                max_new_tokens=128,
            )
        output_text = self.tokenizer.decode(out[0], skip_special_tokens=True)
        print(f'User: {prompt}')
        print(f'Assistant: {output_text}')
        print('*' * 10)
        return output_text[0]
