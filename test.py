# import os
# # from LLMs.LLM import LLM
# from configs.inference_config import get_args
# import pandas as pd
# import csv
# # import torch
# # import torch.nn.functional as F
# import t2v_metrics
# from utils.data.physical_sensations import SENSATIONS_PARENT_MAP
#
# def sequence_logprob(model, tokenizer, phrase: str, context: str = ""):
#     """
#     Compute total log-probability of `phrase` under the model given `context`.
#     Robust to Llama tokenization by explicitly concatenating token IDs.
#     Returns (total_logprob, per_token_logprobs).
#
#     NOTE: If your fine-tune expects chat formatting, make sure `context` already
#     contains the proper template text (e.g., via tokenizer.apply_chat_template).
#     """
#     # Encode without special tokens so lengths add up exactly
#     ctx_ids    = tokenizer.encode(context, add_special_tokens=False)
#     phrase_ids = tokenizer.encode(phrase,  add_special_tokens=False)
#
#     # Prepend BOS if available (common for Llama)
#     prefix_ids = [tokenizer.bos_token_id] if tokenizer.bos_token_id is not None else []
#
#     # Build the exact input the model will see
#     ids = prefix_ids + ctx_ids + phrase_ids
#     input_ids = torch.tensor([ids], device=model.device)
#
#     with torch.no_grad():
#         logits = model(input_ids).logits  # [B, T, V]
#
#     # Next-token log-probs for positions 0..T-2 predicting tokens 1..T-1
#     log_probs = torch.log_softmax(logits[:, :-1, :], dim=-1)
#
#     # Phrase starts after prefix + context
#     offset = len(prefix_ids) + len(ctx_ids)
#
#     per_token = []
#     total_logp = 0.0
#     for r, tok in enumerate(phrase_ids):
#         pos = offset + r  # index in input_ids (token we want probability for)
#         lp = log_probs[0, pos-1, tok].item()  # predicted at previous position
#         per_token.append(lp)
#         total_logp += lp
#
#     return total_logp, per_token
#
# if __name__ == '__main__':
#     args = get_args()
#     # pipe = LLM(args)  # expects attributes: pipe.model.model (HF model), pipe.model.tokenizer
#     #
#     # # Load descriptions
#     clip_flant5_score = t2v_metrics.VQAScore(model='clip-flant5-xxl')  # our recommended scoring model
#     df = pd.read_csv(args.description_file)
#
#     # Prepare output path
#     out_dir = os.path.join(args.result_path, 'results', args.project_name)
#     os.makedirs(out_dir, exist_ok=True)
#     # out_csv = os.path.join(out_dir, f'sensations_predicted_by_{args.LLM}.csv')
#     out_csv = os.path.join(out_dir, 'VQA_score.csv')
#     # Open CSV once
#     with open(out_csv, 'w', newline='', encoding='utf-8') as f:
#         writer = csv.writer(f)
#         # writer.writerow(['ID', 'description', 'predicted_sensation', 'total_logprob', 'per_token_logprobs', 'last_token', 'average'])
#         writer.writerow(['ID', 'score'])
#         # Iterate rows
#         for _, row in df.iterrows():
#             ID = row[0]
#             # description = row[1]
#
#             # If your fine-tuned Llama-3 Instruct expects chat formatting, you can build `prompt`
#             # using the chat template instead of a raw string:
#             #
#             # msgs = [
#             #     dict(role="user",
#             #          content=f"Context: Description of an image is {description}\nGiven the description of the image, the sensation that the image evokes is:")
#             # ]
#             # prompt = pipe.model.tokenizer.apply_chat_template(
#             #     msgs, add_generation_prompt=True, tokenize=False
#             # )
#             #
#             # Otherwise, this plain prompt is fine:
#             # prompt = (
#             #     f"Context: Description of an image is {description}\n"
#             #     f"Given the description of the image, the sensation that the image evokes is: "
#             # )
#             image_path = os.path.join(args.data_path, args.test_set_images, ID)
#             for sensation in SENSATIONS_PARENT_MAP.keys():
#                 try:
#                     # total_logprob, selected_logprobs = sequence_logprob(
#                     #     pipe.model.model,
#                     #     pipe.model.tokenizer,
#                     #     phrase=sensation,
#                     #     context=prompt
#                     # )
#                     score = clip_flant5_score(images=[image_path], texts=[sensation])
#                 except Exception as e:
#                     # Log the error for this (ID, sensation) and continue
#                     print(f"[WARN] ID {ID} | sensation '{sensation}' failed: {e}")
#                     # total_logprob, selected_logprobs = float('-inf'), []
#                     score = float('-inf')
#
#                 # print(f"Sensation for image {ID} is '{sensation}' with score (logP={total_logprob:.4f})")
#                 print(f"sensation for image {ID} is {sensation} with score {score.item()}")
#                 # writer.writerow([ID, description, sensation, total_logprob, selected_logprobs, selected_logprobs[-1], sum(selected_logprobs)/len(selected_logprobs)])
#                 writer.writerow([ID, sensation, score.item()])


from diffusers import DiffusionPipeline
import torch

model_name = "Qwen/Qwen-Image"

# Load the pipeline
if torch.cuda.is_available():
    torch_dtype = torch.bfloat16
    device = "cuda"
else:
    torch_dtype = torch.float32
    device = "cpu"

pipe = DiffusionPipeline.from_pretrained(model_name, torch_dtype=torch_dtype)
pipe = pipe.to(device)

positive_magic = {
    "en": ", Ultra HD, 4K, cinematic composition.", # for english prompt
    "zh": ", 超清，4K，电影级构图." # for chinese prompt
}

# Generate image
prompt = '''Generate an advertisement image that evokes sweet taste sensation and coveys the following messages:                                                                                                                                                                 
                                                                                                                                                                                                                                                                                
    - I should use this product because it will taste good.                                                                                                                                                                                                                     
                                                                                                                                                                                                                                                                                
    - I should become a vegan because the celebrity is.                                                                                                                                                                                                                         
                                                                                                                                                                                                                                                                                
    - I SHOULD EAT THIS BECAUSE IT HAS HEALTHY INGREDIENTS THAT WILL ALSO TASTE GOOD. '''

negative_prompt = " " # using an empty string if you do not have specific concept to remove


# Generate with different aspect ratios
aspect_ratios = {
    "1:1": (1328, 1328),
    "16:9": (1664, 928),
    "9:16": (928, 1664),
    "4:3": (1472, 1140),
    "3:4": (1140, 1472),
    "3:2": (1584, 1056),
    "2:3": (1056, 1584),
}

width, height = aspect_ratios["16:9"]

image = pipe(
    prompt=prompt + positive_magic["en"],
    negative_prompt=negative_prompt,
    width=width,
    height=height,
    num_inference_steps=50,
    true_cfg_scale=4.0,
    generator=torch.Generator(device="cuda").manual_seed(42)
).images[0]

image.save("../experiments/generated_images/SensoryAds/example.png")

