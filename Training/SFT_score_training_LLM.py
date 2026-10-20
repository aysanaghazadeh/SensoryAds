from utils.data.SFT_LLM_data import get_train_LLM_SFT_score_Dataloader
from Training.SFT_training_LLM import get_model, get_training_args
from accelerate import PartialState
from datetime import timedelta
from trl import SFTTrainer

RUN_NAME = "mySFTScore"


def train(args):
    # See SFT_training_LLM.train: the process group timeout must be set before
    # anything else touches PartialState.
    PartialState(timeout=timedelta(seconds=7200))
    sft_args = get_training_args(args, run_name=RUN_NAME)
    model, tokenizer = get_model(args)
    # Rows already carry the chat-templated "text" column.
    dataset = get_train_LLM_SFT_score_Dataloader(args, tokenizer=tokenizer)
    # Fixed seed so every process splits the dataset identically under multi-GPU.
    tmp = dataset.train_test_split(test_size=0.1, seed=42)
    trainer = SFTTrainer(
        model=model,
        args=sft_args,
        train_dataset=tmp["train"],
        eval_dataset=tmp["test"],
        processing_class=tokenizer,
    )

    if args.model_checkpoint is not None:
        print("loading checkpoint")
        trainer.train(
            resume_from_checkpoint=args.model_path
            + f"/{RUN_NAME}_{args.LLM}/checkpoint-{args.model_checkpoint}"
        )
    else:
        print("training from scratch")
        trainer.train()
    trainer.save_model(sft_args.output_dir)
