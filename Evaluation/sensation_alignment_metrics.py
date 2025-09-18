import os
from configs.inference_config import get_args
import pandas as pd
import csv
import t2v_metrics
from utils.data.physical_sensations import SENSATIONS_PARENT_MAP


def get_EvoSense_LLM(args, description, sensation):
    from LLMs.LLM import LLM
    import torch
    import torch.nn.functional as F
    def sequence_logprob(model, tokenizer, phrase: str, context: str = ""):
        """
        Compute total log-probability of `phrase` under the model given `context`.
        Returns (total_logprob, per_token_logprobs).

        NOTE: If your fine-tune expects chat formatting, make sure `context` already
        contains the proper template text (e.g., via tokenizer.apply_chat_template).
        """
        # Encode without special tokens so lengths add up exactly
        ctx_ids    = tokenizer.encode(context, add_special_tokens=False)
        phrase_ids = tokenizer.encode(phrase,  add_special_tokens=False)

        # Prepend BOS if available (common for Llama)
        prefix_ids = [tokenizer.bos_token_id] if tokenizer.bos_token_id is not None else []

        # Build the exact input the model will see
        ids = prefix_ids + ctx_ids + phrase_ids
        input_ids = torch.tensor([ids], device=model.device)

        with torch.no_grad():
            logits = model(input_ids).logits  # [B, T, V]

        log_probs = torch.log_softmax(logits[:, :-1, :], dim=-1)

        # Phrase starts after prefix + context
        offset = len(prefix_ids) + len(ctx_ids)

        per_token = []
        total_logp = 0.0
        for r, tok in enumerate(phrase_ids):
            pos = offset + r
            lp = log_probs[0, pos-1, tok].item()  # predicted at previous position
            per_token.append(lp)
            total_logp += lp

        return total_logp, per_token
    pipe = LLM(args)
    msgs = [
        dict(role="user",
             content=f"Context: Description of an image is {description}\nGiven the description of the image, the sensation that the image evokes is:")
    ]
    prompt = pipe.model.tokenizer.apply_chat_template(
        msgs, add_generation_prompt=True, tokenize=False
    )
    try:
        total_logprob, selected_logprobs = sequence_logprob(
            pipe.model.model,
            pipe.model.tokenizer,
            phrase=sensation,
            context=prompt
        )
    except Exception as e:
        # Log the error for this (ID, sensation) and continue
        print(f"sensation '{sensation}' failed: {e}")
        total_logprob, selected_logprobs = float('-inf'), []

    return total_logprob, selected_logprobs, selected_logprobs[-1], sum(selected_logprobs) / len(selected_logprobs)


def get_EvoSense_MLLM(args, image, sensation):
