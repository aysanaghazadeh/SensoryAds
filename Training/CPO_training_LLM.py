from utils.data.CPO_LLM_data import get_train_LLM_CPO_Dataloader
from configs.training_config import get_args
from transformers import DataCollatorForLanguageModeling
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments
import torch
from peft import get_peft_model, LoraConfig, TaskType, prepare_model_for_kbit_training
from trl import CPOConfig, CPOTrainer, ModelConfig, get_peft_config
import os
import transformers
from LLMs.LLM import LLM

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
        output_dir=args.model_path+f'/my_{args.LLM}',
        remove_unused_columns=False,
        per_device_train_batch_size=args.batch_size,
        gradient_checkpointing=True,
        gradient_accumulation_steps=4,
        max_steps=40000,
        learning_rate=args.lr,
        logging_steps=10,
        fp16=True,
        save_strategy="steps",
        save_steps=100,
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
    cpo_config = CPOConfig(beta=0.1,
                           output_dir=args.model_path+f'/my_{args.LLM}',)
    train_dataset = get_train_LLAMA3_CPO_Dataloader(args)
    tmp = train_dataset.train_test_split(test_size=0.1)
    train_dataset = tmp["train"]
    eval_dataset = tmp["test"]
    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)
    trainer = CPOTrainer(
        model,
        args=cpo_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
    )

    # train and save the model
    trainer.train()
    trainer.save_model(cpo_args.output_dir)