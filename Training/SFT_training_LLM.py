from utils.data.CPO_LLM_data import get_train_LLM_CPO_Dataloader
from configs.training_config import get_args
from accelerate import PartialState
from datetime import timedelta
from peft import get_peft_model, LoraConfig, TaskType, prepare_model_for_kbit_training
from trl import SFTConfig, SFTTrainer
import os
from LLMs.LLM import LLM


def get_model(args):

    pipe = LLM(args)
    model = pipe.model.model
    model.gradient_checkpointing_enable()
    model = prepare_model_for_kbit_training(model)
    peft_config = LoraConfig(inference_mode=False,
                             r=8,
                             lora_alpha=16,
                             lora_dropout=0.1,
                             peft_type=TaskType.CAUSAL_LM)
    # One full replica per process (LLAMA3Instruct already placed it on
    # PartialState().process_index) so accelerate can wrap it in DDP; pinning
    # it to a single args.device here, or flagging it model_parallel, would
    # tell the Trainer to skip DDP wrapping entirely.
    model = get_peft_model(model, peft_config)
    print(f'model\'s trainable parameters: {model.print_trainable_parameters()}')
    tokenizer = pipe.model.tokenizer
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return model, tokenizer


def cpo_rows_to_sft(dataset):
    """Turn pairwise CPO rows into plain LM text using the preferred (chosen) completion."""
    return dataset.map(
        lambda ex: {"text": ex["chosen"]},
        remove_columns=[c for c in dataset.column_names if c != "text"],
    )


def get_training_args(args):
    # args.batch_size is the total batch size across GPUs, so split it per process.
    num_processes = PartialState().num_processes
    per_device_train_batch_size = max(1, args.batch_size // num_processes)
    print(f'total batch size: {args.batch_size} over {num_processes} process(es) '
          f'-> per-device batch size: {per_device_train_batch_size}')
    out = args.model_path + f"/mySFT_{args.LLM}"
    training_args = SFTConfig(
        output_dir=out,
        remove_unused_columns=False,
        per_device_train_batch_size=per_device_train_batch_size,
        # Each rank tokenizes the dataset independently before reaching DDP init,
        # so rank 0 can sit in the setup collective for a long time.
        ddp_timeout=7200,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        gradient_accumulation_steps=4,
        max_steps=40000,
        learning_rate=args.lr,
        logging_steps=10,
        fp16=True,
        save_strategy="steps",
        save_steps=1000,
        eval_strategy="steps",
        eval_steps=1000,
        do_eval=True,
        report_to="none",
        logging_dir=os.path.join(args.results, "logs"),
        dataset_text_field="text",
    )
    if not os.path.exists(os.path.join(args.results, "logs")):
        os.makedirs(os.path.join(args.results, "logs"))
    return training_args


def train(args):
    # Must come before anything else touches PartialState: the process group's
    # timeout is fixed when it is created, so SFTConfig's ddp_timeout arrives
    # too late. Each rank tokenizes the dataset independently before DDP init,
    # which overruns the 10 minute default.
    PartialState(timeout=timedelta(seconds=7200))
    sft_args = get_training_args(args)
    model, tokenizer = get_model(args)
    raw = get_train_LLM_CPO_Dataloader(args, tokenizer=tokenizer)
    # Fixed seed so every process splits the dataset identically under multi-GPU.
    tmp = raw.train_test_split(test_size=0.1, seed=42)
    train_dataset = cpo_rows_to_sft(tmp["train"])
    eval_dataset = cpo_rows_to_sft(tmp["test"])
    trainer = SFTTrainer(
        model=model,
        args=sft_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
    )

    if args.model_checkpoint is not None:
        print("loading checkpoint")
        trainer.train(
            resume_from_checkpoint=args.model_path
            + f"/mySFT_{args.LLM}/checkpoint-{args.model_checkpoint}"
        )
    else:
        print("training from scratch")
        trainer.train()
    trainer.save_model(sft_args.output_dir)
