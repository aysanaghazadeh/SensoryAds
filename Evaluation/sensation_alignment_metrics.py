
def get_EvoSense_LLM(args, pipe, description, sensation):
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


def get_EvoSense_MLLM(args, pipe, builder, image, sensations, chunk_size=8):
    """log p(sensation | image) for every sensation, with the exact prompt and chat format the MLLM
    was trained on (Training/Hierarchical_CPO_training_MLLM.py).

    `pipe` is the MLLMs/* wrapper, `image` a path. Returns {sensation: (total, last_token, average)}."""
    import torch
    from PIL import Image
    from utils.data.HierarchicalCPO_MLLM_data import PROMPT
    from utils.data.MLLM_batch import add_answer_mask
    image = Image.open(image).convert('RGB')
    scores = {}
    for start in range(0, len(sensations), chunk_size):
        names = sensations[start:start + chunk_size]
        # Training answers are the lower-case annotation names.
        answers = [name.lower() for name in names]
        batch, answer_lens = builder.build([PROMPT] * len(names), answers, [image], num_copies=len(names))
        batch = add_answer_mask(batch, answer_lens)
        batch = {k: v.to(args.device) for k, v in batch.items()}
        answer_mask = batch.pop('answer_mask')
        with torch.no_grad():
            logits = pipe.model(**batch).logits
        mask = answer_mask[:, 1:]
        targets = batch['input_ids'][:, 1:][mask]
        selected = logits[:, :-1][mask].float()
        token_logps = torch.log_softmax(selected, dim=-1).gather(-1, targets.unsqueeze(-1)).squeeze(-1)
        for name, logps in zip(names, token_logps.split(mask.sum(dim=1).tolist())):
            logps = logps.tolist()
            scores[name] = (sum(logps), logps[-1], sum(logps) / len(logps)) if logps else (
                float('-inf'), float('-inf'), float('-inf'))
    return scores


def get_T2V_score(args, model, image, text):
    try:
        score = model(images=[image], texts=[text])
    except Exception as e:
        # Log the error for this (ID, sensation) and continue
        print(f"[WARN]sensation '{text}' failed: {e}")
        # total_logprob, selected_logprobs = float('-inf'), []
        score = float('-inf')
    return score

def get_MMLM_Judge_Score(args, model, image, text):
    from utils.prompt_engineering.prompt_generation import generate_text_generation_prompt
    from PIL import Image
    # try:
    data = {
        'sensation': text
    }
    prompt = generate_text_generation_prompt(args, data)
    image = Image.open(image)
    output = model(image, prompt)
    score = int(output.split(':')[-1]) / 5
    # except Exception as e:
    #     # Log the error for this (ID, sensation) and continue
    #     print(f"[WARN]sensation '{text}' failed: {e}")
    #     # total_logprob, selected_logprobs = float('-inf'), []
    #     score = float('-inf')
    return score

def get_LLM_Judge_Score(args, model, description, text):
    from utils.prompt_engineering.prompt_generation import generate_text_generation_prompt
    from PIL import Image
    # try:
    data = {
        'sensation': text,
        'description': description
    }
    prompt = generate_text_generation_prompt(args, data)
    output = model(prompt)
    try:
        score = float(output.split(':')[-1]) / 5
    except Exception as e:
        # Log the error for this (ID, sensation) and continue
        print(f"[WARN]sensation '{text}' failed: {e}")
        # total_logprob, selected_logprobs = float('-inf'), []
        score = float(0)
    # except Exception as e:
    #     # Log the error for this (ID, sensation) and continue
    #     print(f"[WARN]sensation '{text}' failed: {e}")
    #     # total_logprob, selected_logprobs = float('-inf'), []
    #     score = float('-inf')
    return score
