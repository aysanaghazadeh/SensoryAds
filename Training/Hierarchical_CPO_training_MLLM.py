from utils.data.HierarchicalCPO_MLLM_data import get_train_MLLM_HierarchicalCPO_Dataloader
from accelerate import PartialState
from datetime import timedelta
from transformers import Trainer, TrainingArguments
from peft import get_peft_model, LoraConfig, TaskType
from PIL import Image
import torch
import torch.nn.functional as F
import os

from MLLMs.InternVL2 import InternVL
from MLLMs.QWenVL import QWenVL

# Caps image cost per sequence (each row is run three times: chosen, rejected, parent).
QWEN_MAX_IMAGE_SIDE = 448
INTERNVL_MAX_PATCHES = 4
# Language-model projections only; the vision towers ('visual' in Qwen,
# 'vision_model' in InternVL) stay frozen and un-adapted.
LORA_TARGET_REGEX = r"^(?!.*(visual|vision_model)).*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)$"


class QwenBatchBuilder:
    def __init__(self, wrapper):
        self.processor = wrapper.processor
        self.tokenizer = self.processor.tokenizer
        self.tokenizer.padding_side = "right"

    def build(self, prompts, answers, images):
        """prompts/answers cover chosen+rejected+parent (3B); images are the B unique ones."""
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
        batch = dict(self.processor(text=texts, images=images * 3, padding=True, return_tensors="pt"))
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
        self.num_image_token = wrapper.model.num_image_token
        self.system_message = getattr(wrapper.model, "system_message", None)
        # The model scatters image features into the positions of this token.
        wrapper.model.img_context_token_id = self.tokenizer.convert_tokens_to_ids("<IMG_CONTEXT>")

    def build(self, prompts, answers, images):
        pixel_values = [self.wrapper.load_image(image, max_num=INTERNVL_MAX_PATCHES) for image in images]
        num_patches = [pv.shape[0] for pv in pixel_values] * 3
        texts = []
        for prompt, answer, n in zip(prompts, answers, num_patches):
            image_tokens = "<img>" + "<IMG_CONTEXT>" * (self.num_image_token * n) + "</img>"
            text = ""
            if self.system_message:
                text += f"<|im_start|>system\n{self.system_message}<|im_end|>\n"
            text += f"<|im_start|>user\n{image_tokens}\n{prompt}<|im_end|>\n<|im_start|>assistant\n{answer}<|im_end|>\n"
            texts.append(text)
        batch = dict(self.tokenizer(texts, padding=True, return_tensors="pt"))
        pixel_values = torch.cat(pixel_values * 3).to(torch.bfloat16)
        batch["pixel_values"] = pixel_values
        batch["image_flags"] = torch.ones(pixel_values.shape[0], 1, dtype=torch.long)
        answer_lens = [len(self.tokenizer(a + "<|im_end|>\n", add_special_tokens=False).input_ids)
                       for a in answers]
        return batch, answer_lens


class HierarchicalCPOMLLMCollator:
    """Opens the images and builds one concatenated [chosen; rejected; parent] batch of 3B sequences."""

    def __init__(self, builder, image_dir):
        self.builder = builder
        self.image_dir = image_dir

    def __call__(self, rows):
        images = [Image.open(os.path.join(self.image_dir, row["image_url"])).convert("RGB") for row in rows]
        answers = ([row["chosen"] for row in rows]
                   + [row["rejected"] for row in rows]
                   + [row["parent_of_chosen"] for row in rows])
        prompts = [row["prompt"] for row in rows] * 3
        batch, answer_lens = self.builder.build(prompts, answers, images)
        # Right padding: the answer is the last answer_len real tokens of each sequence.
        lengths = batch["attention_mask"].sum(dim=1)
        positions = torch.arange(batch["input_ids"].shape[1]).unsqueeze(0)
        starts = (lengths - torch.tensor(answer_lens)).unsqueeze(1)
        batch["answer_mask"] = (positions >= starts) & (positions < lengths.unsqueeze(1))
        return batch


class HierarchicalCPOMLLMTrainer(Trainer):
    """CPO (sigmoid preference loss + NLL on chosen) plus the hierarchy term of
    HierarchicalCPOTrainer, computed from one forward over [chosen; rejected; parent]."""

    def __init__(self, *args, beta=0.1, cpo_alpha=1.0, hierarchy_loss_weight=1.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.beta = beta
        self.cpo_alpha = cpo_alpha
        self.hierarchy_loss_weight = hierarchy_loss_weight

    @staticmethod
    def answer_logps(logits, input_ids, answer_mask):
        """Summed log-prob of the answer tokens of every sequence, and their counts."""
        mask = answer_mask[:, 1:]
        targets = input_ids[:, 1:][mask]
        # Only answer positions go through the vocab-sized log_softmax.
        selected = logits[:, :-1][mask].float()
        token_logps = torch.log_softmax(selected, dim=-1).gather(-1, targets.unsqueeze(-1)).squeeze(-1)
        rows = torch.arange(input_ids.shape[0], device=input_ids.device).unsqueeze(1).expand_as(mask)[mask]
        sums = torch.zeros(input_ids.shape[0], device=input_ids.device).index_add(0, rows, token_logps)
        return sums, mask.sum(dim=1).clamp(min=1)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        inputs = dict(inputs)
        answer_mask = inputs.pop("answer_mask")
        outputs = model(**inputs)
        sums, counts = self.answer_logps(outputs.logits, inputs["input_ids"], answer_mask)
        chosen_sum, rejected_sum, parent_sum = sums.chunk(3)
        chosen_count, _, parent_count = counts.chunk(3)

        preference_loss = -F.logsigmoid(self.beta * (chosen_sum - rejected_sum)).mean()
        nll_loss = -(chosen_sum / chosen_count).mean()
        # Same as HierarchicalCPOTrainer: ReLU(avg logp(chosen) - avg logp(parent)).
        hierarchy_loss = torch.relu(chosen_sum / chosen_count - parent_sum / parent_count).mean()
        loss = preference_loss + self.cpo_alpha * nll_loss + self.hierarchy_loss_weight * hierarchy_loss
        if self.model.training and self.state.global_step % self.args.logging_steps == 0:
            self.log({"preference_loss": preference_loss.item(),
                      "nll_loss": nll_loss.item(),
                      "hierarchy_loss": hierarchy_loss.item()})
        if return_outputs:
            return loss, outputs
        return loss

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        inputs = self._prepare_inputs(inputs)
        with torch.no_grad():
            loss = self.compute_loss(model, inputs)
        return loss.detach(), None, None


def get_model(args):
    wrappers = {"InternVL": (InternVL, InternVLBatchBuilder), "QWenVL": (QWenVL, QwenBatchBuilder)}
    if args.MLLM not in wrappers:
        raise ValueError(f"Hierarchical CPO MLLM training supports {list(wrappers)}, got MLLM={args.MLLM!r}")
    wrapper_class, builder_class = wrappers[args.MLLM]
    wrapper = wrapper_class(args)
    # Builder first: InternVL's builder sets img_context_token_id on the raw model.
    builder = builder_class(wrapper)
    model = wrapper.model
    peft_config = LoraConfig(inference_mode=False,
                             r=8,
                             lora_alpha=32,
                             lora_dropout=0.1,
                             target_modules=LORA_TARGET_REGEX,
                             task_type=TaskType.CAUSAL_LM)
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    return model, builder


def get_training_args(args):
    # args.batch_size is the total batch size across GPUs, so split it per process.
    num_processes = PartialState().num_processes
    per_device_train_batch_size = max(1, args.batch_size // num_processes)
    print(f'total batch size: {args.batch_size} over {num_processes} process(es) '
          f'-> per-device batch size: {per_device_train_batch_size}')
    training_args = TrainingArguments(
        output_dir=args.model_path + f'/my_HierarchicalCPO_{args.MLLM}',
        # The dataset holds image paths and strings that only the collator understands.
        remove_unused_columns=False,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_train_batch_size,
        ddp_timeout=7200,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        gradient_accumulation_steps=4,
        max_steps=200000,
        learning_rate=args.lr,
        logging_steps=10,
        fp16=True,
        save_strategy="steps",
        save_steps=500,
        eval_strategy="steps",
        eval_steps=100,
        do_eval=True,
        # The collator opens images in the main process; it holds the model-bound processor.
        dataloader_num_workers=0,
        report_to="none",
        logging_dir=os.path.join(args.results, 'logs'),
    )
    if not os.path.exists(os.path.join(args.results, 'logs')):
        os.makedirs(os.path.join(args.results, 'logs'))
    return training_args


def train(args):
    # Must come before anything else touches PartialState: the process group's
    # timeout is fixed when it is created, so TrainingArguments' ddp_timeout
    # arrives too late.
    PartialState(timeout=timedelta(seconds=7200))
    training_args = get_training_args(args)
    model, builder = get_model(args)
    dataset = get_train_MLLM_HierarchicalCPO_Dataloader(args)
    # Fixed seed so every process splits the dataset identically under multi-GPU.
    tmp = dataset.train_test_split(test_size=0.01, seed=42)
    train_dataset = tmp["train"]
    eval_dataset = tmp["test"]
    collator = HierarchicalCPOMLLMCollator(builder, os.path.join(args.data_path, args.test_set_images))
    trainer = HierarchicalCPOMLLMTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collator,
        processing_class=getattr(builder, "processor", builder.tokenizer),
    )

    if args.model_checkpoint is not None:
        print('loading checkpoint')
        trainer.train(resume_from_checkpoint=training_args.output_dir + f'/checkpoint-{args.model_checkpoint}')
    else:
        print('training from scratch')
        trainer.train()
    trainer.save_model(training_args.output_dir)
