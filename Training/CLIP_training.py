from utils.data.HierarchicalCPO_MLLM_data import get_train_MLLM_HierarchicalCPO_Dataloader
from accelerate import PartialState
from datetime import timedelta
from transformers import Trainer, TrainingArguments, CLIPModel, CLIPProcessor
from PIL import Image
import torch
import torch.nn.functional as F
import os

# Same weights as the 'openai:ViT-B-32' model CLIPScore is evaluated with.
CLIP_MODEL_ID = "openai/clip-vit-base-patch32"


class CLIPPairCollator:
    """For every (image, higher-scored sensation, lower-scored sensation) row, builds the image
    batch plus the two text batches. The sensation name is the text, exactly as in CLIPScore."""

    def __init__(self, processor, image_dir):
        self.processor = processor
        self.image_dir = image_dir

    def __call__(self, rows):
        images = [Image.open(os.path.join(self.image_dir, row["image_url"])).convert("RGB") for row in rows]
        texts = [row["chosen"] for row in rows] + [row["rejected"] for row in rows]
        pixel_values = self.processor(images=images, return_tensors="pt")["pixel_values"]
        text_inputs = self.processor(text=texts, padding=True, truncation=True, return_tensors="pt")
        return {"pixel_values": pixel_values,
                "input_ids": text_inputs["input_ids"],
                "attention_mask": text_inputs["attention_mask"]}


class CLIPPreferenceTrainer(Trainer):
    """Pairwise loss: the image must score the sensation annotators rated higher above the
    lower-rated one, with scores being the (temperature-scaled) CLIP cosine similarities."""

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        outputs = model(**inputs)
        batch_size = inputs["pixel_values"].shape[0]
        # logits_per_image is the temperature-scaled cosine similarity of every image with every
        # text; columns [0, B) are the chosen texts and [B, 2B) the rejected ones.
        index = torch.arange(batch_size, device=outputs.logits_per_image.device)
        chosen_scores = outputs.logits_per_image[index, index]
        rejected_scores = outputs.logits_per_image[index, index + batch_size]
        loss = -F.logsigmoid(chosen_scores - rejected_scores).mean()
        if self.model.training and self.state.global_step % self.args.logging_steps == 0:
            self.log({"pair_accuracy": (chosen_scores > rejected_scores).float().mean().item()})
        if return_outputs:
            return loss, outputs
        return loss

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        inputs = self._prepare_inputs(inputs)
        with torch.no_grad():
            loss = self.compute_loss(model, inputs)
        return loss.detach(), None, None


def get_training_args(args):
    num_processes = PartialState().num_processes
    per_device_train_batch_size = max(1, args.batch_size // num_processes)
    print(f'total batch size: {args.batch_size} over {num_processes} process(es) '
          f'-> per-device batch size: {per_device_train_batch_size}')
    training_args = TrainingArguments(
        output_dir=args.model_path + '/my_CLIP',
        remove_unused_columns=False,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_train_batch_size,
        ddp_timeout=7200,
        max_steps=5000,
        # args.lr is an int-typed flag (default 5e-5) and too high for full CLIP fine-tuning.
        learning_rate=1e-5,
        logging_steps=10,
        bf16=True,
        save_strategy="steps",
        save_steps=500,
        eval_strategy="steps",
        eval_steps=250,
        do_eval=True,
        dataloader_num_workers=4,
        report_to="none",
        logging_dir=os.path.join(args.results, 'logs'),
    )
    if not os.path.exists(os.path.join(args.results, 'logs')):
        os.makedirs(os.path.join(args.results, 'logs'))
    return training_args


def train(args):
    PartialState(timeout=timedelta(seconds=7200))
    training_args = get_training_args(args)
    model = CLIPModel.from_pretrained(CLIP_MODEL_ID)
    processor = CLIPProcessor.from_pretrained(CLIP_MODEL_ID)
    dataset = get_train_MLLM_HierarchicalCPO_Dataloader(args)
    # Fixed seed so every process splits the dataset identically under multi-GPU.
    tmp = dataset.train_test_split(test_size=0.01, seed=42)
    eval_dataset = tmp["test"].select(range(min(2000, len(tmp["test"]))))
    trainer = CLIPPreferenceTrainer(
        model=model,
        args=training_args,
        train_dataset=tmp["train"],
        eval_dataset=eval_dataset,
        data_collator=CLIPPairCollator(processor, os.path.join(args.data_path, args.test_set_images)),
        processing_class=processor,
    )
    if args.model_checkpoint is not None:
        print('loading checkpoint')
        trainer.train(resume_from_checkpoint=training_args.output_dir + f'/checkpoint-{args.model_checkpoint}')
    else:
        print('training from scratch')
        trainer.train()
    trainer.save_model(training_args.output_dir)
