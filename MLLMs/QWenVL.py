import os.path

from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor, BitsAndBytesConfig
from qwen_vl_utils import process_vision_info
import torch
from peft import PeftModel

class QWenVL(torch.nn.Module):
    def __init__(self, args):
        super().__init__()
        bnb_config = BitsAndBytesConfig(load_in_8bit=True)
        self.device = args.device
        if not args.fine_tuned:
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct",
                                                                        quantization_config=bnb_config,
                                                                        device_map='auto').eval()
        else:
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                "Qwen/Qwen2.5-VL-7B-Instruct",
                device_map="auto",
                torch_dtype=torch.bfloat16  # or float16 if you used that
            )

            # Load LoRA fine-tuned adapter
            self.model = PeftModel.from_pretrained(self.model, os.path.join(args.model_path, 'checkpoint-1500'))
        # self.model = self.model.to(device=args.device)


        self.processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")


    def forward(self, image, prompt, generate_kwargs):

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": image,
                    },
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        # Preparation for inference
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to("cuda")

        # Inference: Generation of the output
        generated_ids = self.model.generate(**inputs,
                                            max_new_tokens=generate_kwargs['max_new_tokens'])
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        print(f'User: {prompt}')
        print(f'Assistant: {output_text}')
        print('*' * 10)
        return output_text[0]
