import os
from LLMs.LLM import LLM
from configs.inference_config import get_args
import pandas as pd
import csv
import torch
import torch.nn.functional as F
from utils.data.physical_sensations import SENSATIONS_PARENT_MAP

def sequence_logprob(model, tokenizer, phrase: str, context: str = ""):
    """
    Compute total log-probability of `phrase` under the model given `context`.
    Robust to Llama tokenization by explicitly concatenating token IDs.
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

    # Next-token log-probs for positions 0..T-2 predicting tokens 1..T-1
    log_probs = torch.log_softmax(logits[:, :-1, :], dim=-1)

    # Phrase starts after prefix + context
    offset = len(prefix_ids) + len(ctx_ids)

    per_token = []
    total_logp = 0.0
    for r, tok in enumerate(phrase_ids):
        pos = offset + r  # index in input_ids (token we want probability for)
        lp = log_probs[0, pos-1, tok].item()  # predicted at previous position
        per_token.append(lp)
        total_logp += lp

    return total_logp, per_token

if __name__ == '__main__':
    args = get_args()
    pipe = LLM(args)  # expects attributes: pipe.model.model (HF model), pipe.model.tokenizer

    # Load descriptions
    df = pd.read_csv(args.description_file)

    # Prepare output path
    out_dir = os.path.join(args.result_path, 'results', args.project_name)
    os.makedirs(out_dir, exist_ok=True)
    out_csv = os.path.join(out_dir, f'sensations_predicted_by_{args.LLM}.csv')

    # Open CSV once
    with open(out_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['ID', 'description', 'predicted_sensation', 'total_logprob', 'per_token_logprobs'])

        # Iterate rows
        for _, row in df.iterrows():
            ID = row[0]
            description = row[1]

            # If your fine-tuned Llama-3 Instruct expects chat formatting, you can build `prompt`
            # using the chat template instead of a raw string:
            #
            msgs = [
                dict(role="user",
                     content=f"Context: Description of an image is {description}\nGiven the description of the image, the sensation that the image evokes is:")
            ]
            prompt = pipe.model.tokenizer.apply_chat_template(
                msgs, add_generation_prompt=True, tokenize=False
            )
            #
            # Otherwise, this plain prompt is fine:
            # prompt = (
            #     f"Context: Description of an image is {description}\n"
            #     f"Given the description of the image, the sensation that the image evokes is: "
            # )

            for sensation in SENSATIONS_PARENT_MAP.keys():
                try:
                    total_logprob, selected_logprobs = sequence_logprob(
                        pipe.model.model,
                        pipe.model.tokenizer,
                        phrase=sensation,
                        context=prompt
                    )
                except Exception as e:
                    # Log the error for this (ID, sensation) and continue
                    print(f"[WARN] ID {ID} | sensation '{sensation}' failed: {e}")
                    total_logprob, selected_logprobs = float('-inf'), []

                print(f"Sensation for image {ID} is '{sensation}' with score (logP={total_logprob:.4f})")
                writer.writerow([ID, description, sensation, total_logprob, selected_logprobs])
