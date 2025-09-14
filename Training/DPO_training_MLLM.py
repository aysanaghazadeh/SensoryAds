from datasets import features, load_dataset
from transformers import AutoModelForVision2Seq, AutoProcessor, Qwen2_5_VLForConditionalGeneration
import torch
from trl import DPOConfig, DPOTrainer
from peft import get_peft_model, LoraConfig, TaskType, prepare_model_for_kbit_training
from utils.data.DPO_MLLM_data import get_train_MLLM_DPO_Dataloader
from configs.training_config import get_args

args = get_args()
dataset = get_train_MLLM_DPO_Dataloader(args)
model = Qwen2_5_VLForConditionalGeneration.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct",
                                                                device_map='auto').eval()
model.gradient_checkpointing_enable()
model = prepare_model_for_kbit_training(model)
peft_config = LoraConfig(inference_mode=False,
                          r=8,
                          lora_alpha=32,
                          lora_dropout=0.1,
                          target_modules=["q_proj", "v_proj"],
                          peft_type=TaskType.CAUSAL_LM)
model = get_peft_model(model, peft_config).to(device='cuda')
print(f'model\'s trainable parameters: {model.print_trainable_parameters()}')
if torch.cuda.device_count() > 1:
    print(f'torch cuda count: {torch.cuda.device_count()}')
    model.is_parallelizable = True
    model.model_parallel = True
processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")
print(dataset)
def format(example):
    # Prepare the input for the chat template
    prompt = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": example["question"]}]}]
    chosen = [{"role": "assistant", "content": [{"type": "text", "text": example["chosen"]}]}]
    rejected = [{"role": "assistant", "content": [{"type": "text", "text": example["rejected"]}]}]
    # Apply the chat template
    prompt = processor.apply_chat_template(prompt, tokenize=False)
    chosen = processor.apply_chat_template(chosen, tokenize=False)
    rejected = processor.apply_chat_template(rejected, tokenize=False)
    # Resize the image to ensure it fits within the maximum allowable
    # size of the processor to prevent OOM errors.
    max_size = processor.image_processor.size["longest_edge"] // 2
    example["image"].thumbnail((max_size, max_size))
    return {"images": [example["image"]], "prompt": prompt, "chosen": chosen, "rejected": rejected}

# Apply the formatting function to the dataset
dataset = dataset.map(format, remove_columns=dataset.column_names)

# Make sure that the images are decoded, it prevents from storing bytes.
# More info here https://github.com/huggingface/blog/pull/2148#discussion_r1667400478
f = dataset.features
f["images"] = features.Sequence(features.Image(decode=True))
dataset = dataset.cast(f)

# Train the model
training_args = DPOConfig(
    output_dir=f"my_{args.MLLM}",
    bf16=True,
    gradient_checkpointing=True,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=32,
    num_train_epochs=10,
    # dataset_num_proc=32,  # tokenization will use 32 processes
    # dataloader_num_workers=32,  # data loading will use 32 workers
    logging_steps=300,
)
trainer = DPOTrainer(
    model,
    ref_model=None,  # not needed when using peft
    args=training_args,
    train_dataset=dataset,
    processing_class=processor,
    peft_config=LoraConfig(target_modules="all-linear"),
)

trainer.train()

