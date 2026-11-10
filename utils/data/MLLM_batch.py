import torch

# Caps image cost per sequence (each training row is run three times: chosen, rejected, parent).
QWEN_MAX_IMAGE_SIDE = 448
INTERNVL_MAX_PATCHES = 4


class QwenBatchBuilder:
    def __init__(self, wrapper):
        self.processor = wrapper.processor
        self.tokenizer = self.processor.tokenizer
        self.tokenizer.padding_side = "right"

    def build(self, prompts, answers, images, num_copies=3):
        """prompts/answers cover num_copies groups of len(images) sequences each; the groups
        share the same images (training: chosen, rejected, parent; evaluation: one image, many answers)."""
        images = [image.copy() for image in images]
        for image in images:
            image.thumbnail((QWEN_MAX_IMAGE_SIDE, QWEN_MAX_IMAGE_SIDE))
        texts = []
        for prompt, answer in zip(prompts, answers):
            messages = [
                {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]},
                {"role": "assistant", "content": [{"type": "text", "text": answer}]},
            ]
            texts.append(self.processor.apply_chat_template(messages, tokenize=False))
        batch = dict(self.processor(text=texts, images=images * num_copies, padding=True, return_tensors="pt"))
        answer_lens = [len(self.tokenizer(a + "<|im_end|>\n", add_special_tokens=False).input_ids)
                       for a in answers]
        return batch, answer_lens


class InternVLBatchBuilder:
    def __init__(self, wrapper):
        self.wrapper = wrapper
        self.tokenizer = wrapper.tokenizer
        self.tokenizer.padding_side = "right"
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        # A fine-tuned wrapper holds a PeftModel; the image-token attribute belongs on the raw model.
        base_model = wrapper.model.get_base_model() if hasattr(wrapper.model, "get_base_model") else wrapper.model
        self.num_image_token = base_model.num_image_token
        self.system_message = getattr(base_model, "system_message", None)
        # The model scatters image features into the positions of this token.
        base_model.img_context_token_id = self.tokenizer.convert_tokens_to_ids("<IMG_CONTEXT>")

    def build(self, prompts, answers, images, num_copies=3):
        pixel_values = [self.wrapper.load_image(image, max_num=INTERNVL_MAX_PATCHES) for image in images]
        num_patches = [pv.shape[0] for pv in pixel_values] * num_copies
        texts = []
        for prompt, answer, n in zip(prompts, answers, num_patches):
            image_tokens = "<img>" + "<IMG_CONTEXT>" * (self.num_image_token * n) + "</img>"
            text = ""
            if self.system_message:
                text += f"<|im_start|>system\n{self.system_message}<|im_end|>\n"
            text += f"<|im_start|>user\n{image_tokens}\n{prompt}<|im_end|>\n<|im_start|>assistant\n{answer}<|im_end|>\n"
            texts.append(text)
        batch = dict(self.tokenizer(texts, padding=True, return_tensors="pt"))
        pixel_values = torch.cat(pixel_values * num_copies).to(torch.bfloat16)
        batch["pixel_values"] = pixel_values
        batch["image_flags"] = torch.ones(pixel_values.shape[0], 1, dtype=torch.long)
        answer_lens = [len(self.tokenizer(a + "<|im_end|>\n", add_special_tokens=False).input_ids)
                       for a in answers]
        return batch, answer_lens


def get_batch_builder(wrapper, MLLM):
    builders = {"InternVL": InternVLBatchBuilder, "QWenVL": QwenBatchBuilder}
    if MLLM not in builders:
        raise ValueError(f"supported MLLMs are {list(builders)}, got MLLM={MLLM!r}")
    return builders[MLLM](wrapper)


def add_answer_mask(batch, answer_lens):
    """Right padding: the answer is the last answer_len real tokens of each sequence."""
    lengths = batch["attention_mask"].sum(dim=1)
    positions = torch.arange(batch["input_ids"].shape[1]).unsqueeze(0)
    starts = (lengths - torch.tensor(answer_lens)).unsqueeze(1)
    batch["answer_mask"] = (positions >= starts) & (positions < lengths.unsqueeze(1))
    return batch
