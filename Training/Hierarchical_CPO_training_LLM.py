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


# 1. Define the Custom Trainer with Hierarchical Loss
# ----------------------------------------------------

class HierarchicalCPOTrainer(CPOTrainer):
    """
    A CPOTrainer that incorporates a hierarchical preference loss.

    This trainer assumes the preference is: parent_of_chosen > chosen > rejected.
    It adds a second CPO-style loss term for the (parent, chosen) pair.
    """

    def __init__(self, *args, hierarchy_loss_weight: float = 0.5, **kwargs):
        super().__init__(*args, **kwargs)
        self.hierarchy_loss_weight = hierarchy_loss_weight
        print(f"HierarchicalCPOTrainer initialized with hierarchy_loss_weight = {self.hierarchy_loss_weight}")

    def compute_loss(
            self,
            model: torch.nn.Module,
            inputs: dict,
            return_outputs: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict]:
        """
        Computes the CPO loss plus the hierarchical preference loss.
        """
        if not self.use_dpo_data_collator:
            raise ValueError("HierarchicalCPOTrainer requires the CPODataCollator.")

        # --- Standard CPO Loss Calculation ---
        # Get log probabilities for the (chosen, rejected) pair
        policy_chosen_logps, policy_rejected_logps, _, _ = self.get_batch_logps(
            model,
            inputs,
            average_log_prob=self.loss_config.average_log_prob,
        )

        # Calculate the standard CPO loss term
        # (This is a simplified representation of the CPO loss logic)
        logits = policy_chosen_logps - policy_rejected_logps
        if self.loss_config.loss_type == "ipo":
            base_loss = (logits - 1 / (2 * self.beta)).pow(2).mean()
        else:  # sigmoid, hinge, etc. handled by the parent class's loss function
            # For simplicity, we'll call the parent's loss function
            # This is more robust as it respects the exact loss_type from the config
            base_loss, _ = super().loss(policy_chosen_logps, policy_rejected_logps, None, None)

        # --- Hierarchical Preference Loss Calculation ---

        # Create a temporary input dictionary for the hierarchical pair
        hierarchy_inputs = {
            "prompt": inputs["prompt"],
            "prompt_input_ids": inputs["prompt_input_ids"],
            "prompt_attention_mask": inputs["prompt_attention_mask"],
            # The "parent" is the new "chosen"
            "chosen_input_ids": inputs["parent_of_chosen_input_ids"],
            "chosen_attention_mask": inputs["parent_of_chosen_attention_mask"],
            "chosen_labels": inputs["parent_of_chosen_labels"],
            # The original "chosen" is the new "rejected"
            "rejected_input_ids": inputs["chosen_input_ids"],
            "rejected_attention_mask": inputs["chosen_attention_mask"],
            "rejected_labels": inputs["chosen_labels"],
        }

        policy_parent_logps, policy_chosen_for_hierarchy_logps, _, _ = self.get_batch_logps(
            model,
            hierarchy_inputs,
            average_log_prob=self.loss_config.average_log_prob,
        )

        # Calculate the hierarchical loss term using the same CPO/DPO-style loss\
        # method could also be used if it were refactored to be more modular.
        # This enforces: logp(parent) >= logp(chosen)
        # The loss is max(0, logp_chosen - logp_parent)

        logp_difference = policy_chosen_for_hierarchy_logps - policy_parent_logps
        hierarchical_loss = F.relu(logp_difference).mean()
        # --- Combine the Losses ---
        total_loss = base_loss + self.hierarchy_loss_weight * hierarchical_loss

        # The rest is for metrics and output handling, mirroring the parent class
        chosen_rewards = self.beta * policy_chosen_logps.detach()
        rejected_rewards = self.beta * policy_rejected_logps.detach()
        parent_rewards = self.beta * policy_parent_logps.detach()

        metrics = {}
        metrics["loss/base"] = base_loss.item()
        metrics["loss/hierarchy"] = hierarchical_loss.item()
        metrics["loss/total"] = total_loss.item()
        metrics["rewards/chosen"] = chosen_rewards.mean().item()
        metrics["rewards/rejected"] = rejected_rewards.mean().item()
        metrics["rewards/parent"] = parent_rewards.mean().item()
        metrics["rewards/margins"] = (chosen_rewards - rejected_rewards).mean().item()
        metrics["rewards/hierarchy_margins"] = (parent_rewards - chosen_rewards).mean().item()
        metrics["logps/chosen"] = policy_chosen_logps.detach().mean().item()
        metrics["logps/rejected"] = policy_rejected_logps.detach().mean().item()
        metrics["logps/parent"] = policy_parent_logps.detach().mean().item()

        # The `push_to_hub` method requires the metrics to be part of the outputs
        outputs = {"loss_metrics": metrics}

        if return_outputs:
            return total_loss, outputs

        return total_loss


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