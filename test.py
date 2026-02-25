# import os
# # from LLMs.LLM import LLM
# from configs.inference_config import get_args
# import pandas as pd
# import csv
# import torch
# import torch.nn.functional as F
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


# import json
# file = json.load(open('/Users/aysanaghazadeh/experiments/SensoryAds/IN_InternVL_train_images_total_ALL_description_generation.json'))
# data = file['10/177623.png']
# sorted_dict = dict(sorted(data.items(), key=lambda item: item[1][2], reverse=True))
# print(sorted_dict)

# import json

# AIM = json.load(open('/Users/aysanaghazadeh/IN_InternVL_20250916_122348_AR_ALL_Flux_ALL_description_generationLLAMA3_instruct_text_image_alignment_isFineTunedTrue_3000_weighted.json'))
# scores_sum = 0
# count = 0
# max, min = -float('inf'), float('inf')
# for image_url in AIM:
#     if count > 500:
#         continue
    
#     if isinstance(AIM[image_url], list):
#         score = AIM[image_url][1]
#         if score > max:
#             max = score
#         if score < min:
#             min = score
#         scores_sum += score
#         count += 1
# print(scores_sum / count)
# print(max, min)

import json
import os

filenames = os.listdir('/Users/aysanaghazadeh/SensoryAds')
maximum, minimum = -float('inf'), float('inf')
for filename in filenames:
    if 'isFineTunedTrue_3000_weighted.json' in filename:
        AIM = json.load(open(os.path.join('/Users/aysanaghazadeh/SensoryAds', filename)))
        for image_url in AIM:
            if isinstance(AIM[image_url], list):
                score = AIM[image_url][1]
                if score > maximum:
                    maximum = score
                if score < minimum:
                    minimum = score
print(maximum, minimum)
for filename in filenames:
    if 'isFineTunedTrue_3000_weighted.json' in filename and 'QWenVL' not in filename:
        AIM = json.load(open(os.path.join('/Users/aysanaghazadeh/SensoryAds', filename)))
        sum_scores = 0
        count = 0
        for image_url in AIM:
            if isinstance(AIM[image_url], list):
                score = AIM[image_url][1]
                sum_scores += score#(score - minimum) / (maximum - minimum)
                count += 1
        print(filename)
        print(sum_scores / count)
        print(count)
        print('--------------------------------')

# filenames = os.listdir('/Users/aysanaghazadeh/SensoryAds')
# maximum, minimum = -float('inf'), float('inf')
# for filename in filenames:
#     if 'persuasion.json' in filename:
#         persuasion = json.load(open(os.path.join('/Users/aysanaghazadeh/SensoryAds', filename)))
#         for image_url in persuasion:
#             if isinstance(persuasion[image_url], list):
#                 score = persuasion[image_url][-1] / 5
#                 if score > maximum:
#                     maximum = score
#                 if score < minimum:
#                     minimum = score
# print(maximum, minimum)
# for filename in filenames:
#     if 'persuasion.json' in filename:
#         persuasion = json.load(open(os.path.join('/Users/aysanaghazadeh/SensoryAds', filename)))
#         sum_scores = 0
#         count = 0
#         for image_url in persuasion:
#             if count > 303:
#                 break
#             if isinstance(persuasion[image_url], list):
#                 score = persuasion[image_url][-1] / 5
#                 # sum_scores += (score - minimum) / (maximum - minimum)
#                 sum_scores += score
#                 count += 1
#         print(filename)
#         print(sum_scores / count)
#         print(count)
#         print('--------------------------------')
        

# import json
# import pandas as pd
# human_scores = json.load(open('../Data/PittAd/train/sensation_annotations_parsed.json'))
# found_sensations = pd.read_csv('/Users/aysanaghazadeh/Downloads/sensation_found.csv')
# found_sensations = found_sensations.values

# correct_count = 0
# total_count = 0
# for row in found_sensations:
#     image_url = row[0].split('-')[0]
#     found_sensation = row[1]
#     if image_url not in human_scores:
#         print(image_url)
#         continue
#     if found_sensation.lower() in human_scores[image_url]['sensation_scores']:
#         if human_scores[image_url]['sensation_scores'][found_sensation.lower()] > 0:
#             print(image_url)
#             correct_count += 1
#         # else:
#             # print(image_url)
#             # print(found_sensation)
#         for sensation in human_scores[image_url]['sensation_scores']:
#             if human_scores[image_url]['sensation_scores'][sensation] > 0:
#                 total_count += 1
#     else:
#         print(image_url)
#         print(found_sensation)
# print(correct_count)
# print(correct_count / total_count)
# print(total_count)
# import json
# scores = 0
# count = 0
# persuasion =  json.load(open('/Users/aysanaghazadeh/IN_InternVL_20260129_002256_AR_ALL_AgenticEditing_ALL_description_generation_LLAMA3_instruct_persuasion.json'))
# aim = json.load(open('/Users/aysanaghazadeh/IN_InternVL_20260129_002256_AR_ALL_AgenticEditing_ALL_description_generationLLAMA3_instruct_text_image_alignment_isFineTunedTrue_3000_weighted.json'))
# for image_url in persuasion:
#     scores += (persuasion[image_url][-1] * (len(persuasion[image_url]) - 2) / 5 + sum(aim[image_url][-1])/len(aim[image_url][-1])) / (len(persuasion[image_url]) - 1)
#     count += 1
# print(scores / count)
# print(count)
# scores = 0
# count = 0
# persuasion =  json.load(open('/Users/aysanaghazadeh/IN_InternVL_20260205_011533_AR_ALL_AgenticEditing_ALL_description_generation_LLAMA3_instruct_persuasion.json'))
# aim = json.load(open('/Users/aysanaghazadeh/IN_InternVL_20260205_011533_AR_ALL_AgenticEditing_ALL_description_generationLLAMA3_instruct_text_image_alignment_isFineTunedTrue_3000_weighted.json'))
# for image_url in persuasion:
#     scores += (persuasion[image_url][-1] * (len(persuasion[image_url]) - 2) / 5 + sum(aim[image_url][-1])/len(aim[image_url][-1])) / (len(persuasion[image_url]) - 1)
#     count += 1
# print(scores / count)
# print(count)
