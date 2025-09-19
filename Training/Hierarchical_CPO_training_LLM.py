from utils.data.HierarchicalCPO_LLM_data import get_train_LLM_HierarchicalCPO_Dataloader
from configs.training_config import get_args
from transformers import DataCollatorForLanguageModeling
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments
import torch
from peft import get_peft_model, LoraConfig, TaskType, prepare_model_for_kbit_training
from trl import CPOConfig, CPOTrainer, ModelConfig, get_peft_config
import os
import transformers
from LLMs.LLM import LLM

import torch
import torch.nn.functional as F
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from trl import CPOTrainer, CPOConfig
import torch
import torch.nn.functional as F
from transformers import PreTrainedModel
from torch import nn
from typing import Union, Any
from contextlib import nullcontext

# For mixed-precision training
try:
    from torch.cuda.amp import autocast
except ImportError:
    class autocast:
        def __init__(self, enabled=True):
            pass

        def __enter__(self):
            pass

        def __exit__(self, exc_type, exc_val, exc_tb):
            pass


# We will edit your provided class structure
class HierarchicalCPOTrainer(CPOTrainer):
    def __init__(self, *args, hierarchy_loss_weight: float = 0.5, **kwargs):
        super().__init__(*args, **kwargs)
        self.hierarchy_loss_weight = hierarchy_loss_weight

    def compute_loss(
            self,
            model: Union[PreTrainedModel, nn.Module],
            inputs: dict[str, Union[torch.Tensor, Any]],
            return_outputs=False,
            num_items_in_batch=None,
    ) -> Union[torch.Tensor, tuple[torch.Tensor, dict[str, torch.Tensor]]]:
        compute_loss_context_manager = (
            autocast(self.accelerator.device.type) if self._peft_has_been_casted_to_bf16 else nullcontext()
        )
        with compute_loss_context_manager:
            loss, metrics = self.get_batch_loss_metrics(model, inputs, train_eval="train")
        (chosen_logps, rejected_logps, chosen_logits, rejected_logits, nll_loss) = self.concatenated_forward(model, inputs)
        print(chosen_logps, rejected_logps)
        (parent_logps, chosen_logps, parent_logits, chosen_logits, nll_loss) = self.concatenated_forward(model,
                                                                                                         batch={
                                                                                                             "prompt_input_ids":inputs["prompt_input_ids"],
                                                                                                             "prompt_attention_mask":inputs["prompt_attention_mask"],
                                                                                                             'chosen_input_ids': inputs['parent_input_ids'],
                                                                                                             'chosen_attention_mask': inputs['parent_attention_mask'],
                                                                                                             'rejected_input_ids': inputs['chosen_input_ids'],
                                                                                                             'rejected_attention_mask': inputs['chosen_attention_mask']
                                                                                                         })
        print(parent_logps, chosen_logps)
        print('-'*100)
        # force log the metrics
        self.store_metrics(metrics, train_eval="train")

        if return_outputs:
            return (loss, metrics)
        return loss


def get_model(args):

    pipe = LLM(args)
    model = pipe.model.model
    model.gradient_checkpointing_enable()
    model = prepare_model_for_kbit_training(model)
    peft_config = LoraConfig(inference_mode=False,
                             r=8,
                             lora_alpha=32,
                             lora_dropout=0.1,
                             peft_type=TaskType.CAUSAL_LM)
    model = get_peft_model(model, peft_config).to(device=args.device)
    print(f'model\'s trainable parameters: {model.print_trainable_parameters()}')
    if torch.cuda.device_count() > 1:
        print(f'torch cuda count: {torch.cuda.device_count()}')
        model.is_parallelizable = True
        model.model_parallel = True
    tokenizer = pipe.model.tokenizer
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return model, tokenizer


def get_training_args(args):
    training_args = CPOConfig(
        output_dir=args.model_path+f'/my_HierarchicalCPO_{args.LLM}',
        remove_unused_columns=False,
        per_device_train_batch_size=args.batch_size,
        gradient_checkpointing=True,
        gradient_accumulation_steps=4,
        max_steps=40000,
        learning_rate=args.lr,
        logging_steps=10,
        fp16=True,
        save_strategy="steps",
        save_steps=500,
        eval_strategy="steps",
        eval_steps=100,
        do_eval=True,
        label_names=["input_ids", "labels", "attention_mask"],
        report_to="none",
        logging_dir=os.path.join(args.results, 'logs')
    )
    if not os.path.exists(os.path.join(args.results, 'logs')):
        os.makedirs(os.path.join(args.results, 'logs'))
    return training_args


def train(args):
    cpo_args = get_training_args(args)
    model, tokenizer = get_model(args)
    train_dataset = get_train_LLM_HierarchicalCPO_Dataloader(args, tokenizer)
    tmp = train_dataset.train_test_split(test_size=0.1)
    train_dataset = tmp["train"]

    eval_dataset = tmp["test"]
    trainer = HierarchicalCPOTrainer(
        model,
        args=cpo_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
    )

    # train and save the model
    trainer.train()
    trainer.save_model(cpo_args.output_dir)