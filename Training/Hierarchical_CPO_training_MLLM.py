from utils.data.HierarchicalCPO_MLLM_data import get_train_MLLM_HierarchicalCPO_Dataloader
from accelerate import PartialState
from datetime import timedelta
from transformers import Trainer, TrainingArguments
from peft import get_peft_model, LoraConfig
from PIL import Image
import torch
import torch.nn.functional as F
import os

from MLLMs.InternVL2 import InternVL
from MLLMs.QWenVL import QWenVL
from utils.data.MLLM_batch import get_batch_builder, add_answer_mask

# Language-model projections only; the vision towers ('visual' in Qwen,
# 'vision_model' in InternVL) stay frozen and un-adapted.
LORA_TARGET_REGEX = r"^(?!.*(visual|vision_model)).*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)$"


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
        return add_answer_mask(batch, answer_lens)


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
    wrappers = {"InternVL": InternVL, "QWenVL": QWenVL}
    if args.MLLM not in wrappers:
        raise ValueError(f"Hierarchical CPO MLLM training supports {list(wrappers)}, got MLLM={args.MLLM!r}")
    wrapper = wrappers[args.MLLM](args)
    # Builder first: InternVL's builder sets img_context_token_id on the raw model.
    builder = get_batch_builder(wrapper, args.MLLM)
    model = wrapper.model
    peft_config = LoraConfig(inference_mode=False,
                             r=8,
                             lora_alpha=32,
                             lora_dropout=0.1,
                             target_modules=LORA_TARGET_REGEX)
    # No task_type on purpose: that selects PeftModelForCausalLM, whose forward always passes
    # inputs_embeds/labels/... to the base model, and InternVLChatModel.forward rejects them.
    # The generic PeftModel forwards exactly the kwargs the collator builds.
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
