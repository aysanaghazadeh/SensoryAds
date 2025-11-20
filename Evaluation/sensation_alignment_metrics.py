
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


def get_EvoSense_MLLM(args, pipe, image, sensation):
    import torch
    import torch.nn.functional as F
    image = Image.open(image)
    user_msgs = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": f"Given the image, the sensation that the image evokes is:"}
            ],
        }
    ]
    context_text = pipe.processor.apply_chat_template(user_msgs, add_generation_prompt=True, tokenize=False)

    # Tokenize target continuation separately (no special tokens)
    target_ids = pipe.processor.tokenizer.encode(sensation, add_special_tokens=False)

    # 1) Encode context (image + text), run once to get past_key_values and last logits
    ctx_inputs = pipe.processor(images=image, text=context_text, return_tensors="pt")
    ctx_inputs = {k: v.to(args.device) for k, v in ctx_inputs.items()}

    with torch.no_grad():
        out = pipe.model(**ctx_inputs, use_cache=True)
        past = out.past_key_values
        last_logits = out.logits[:, -1, :]  # predicts the first target token

    # 2) Incrementally consume target tokens to accumulate per-token logprobs
    per_token_logprobs = []
    total_logprob = 0.0

    for tok in target_ids:
        # log p(tok | context + previous target tokens)
        lp = F.log_softmax(last_logits, dim=-1)[0, tok].item()
        per_token_logprobs.append(lp)
        total_logprob += lp

        # advance the state by feeding the actual token
        tok_tensor = torch.tensor([[tok]], device=args.device)
        with torch.no_grad():
            step_out = pipe.model(input_ids=tok_tensor, past_key_values=past, use_cache=True)
        past = step_out.past_key_values
        last_logits = step_out.logits[:, -1, :]

    # Safety for empty target
    if len(per_token_logprobs) == 0:
        return float("-inf"), [], float("-inf"), float("-inf")

    return (
        total_logprob,
        per_token_logprobs,
        per_token_logprobs[-1],
        sum(per_token_logprobs) / len(per_token_logprobs),
    )


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
    # try:
    data = {
        'sensation': text
    }
    prompt = generate_text_generation_prompt(args, data)
    output = model(image, prompt)
    score = int(output.split(':')[-1]) / 5
    # except Exception as e:
    #     # Log the error for this (ID, sensation) and continue
    #     print(f"[WARN]sensation '{text}' failed: {e}")
    #     # total_logprob, selected_logprobs = float('-inf'), []
    #     score = float('-inf')
    return score
