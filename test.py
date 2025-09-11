import os
from LLMs.LLM import LLM
from configs.inference_config import get_args
import pandas as pd
import csv
import torch
from utils.data.physical_sensations import SENSATIONS_PARENT_MAP

if __name__ == '__main__':
    def sequence_logprob(model, tokenizer, phrase, context):
        input_text = context + phrase
        enc = tokenizer(input_text, return_tensors="pt")
        input_ids = enc["input_ids"].to(model.device)

        with torch.no_grad():
            outputs = model(input_ids)
            logits = outputs.logits

        shift_logits = logits[:, :-1, :]
        shift_labels = input_ids[:, 1:]

        log_probs = torch.nn.functional.log_softmax(shift_logits, dim=-1)

        phrase_ids = tokenizer(phrase, return_tensors="pt")["input_ids"][0]
        context_ids = tokenizer(context, return_tensors="pt")["input_ids"][0]
        start = len(context_ids)
        end = start + len(phrase_ids)

        selected_log_probs = []
        for i in range(start, end):
            token_id = input_ids[0, i]
            lp = log_probs[0, i - 1, token_id].item()
            selected_log_probs.append(lp)

        total_logprob = sum(selected_log_probs)
        return total_logprob, selected_log_probs
    args = get_args()
    pipe = LLM(args)
    descriptions = pd.read_csv(args.description_file).values
    with open(os.path.join(args.result_path, 'results', args.project_name, f'sensations_predicted_by_{args.LLM}.csv'), 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['ID', 'description', 'predicted_sensation', 'total_logprob', 'selected_log_probs'])
    for row in descriptions:
        ID = row[0]
        description = row[1]
        prompt = f"""Context: Description of an image is {description}
            Given the description of the image, the sensation that the image evokes is: """

        for sensation in SENSATIONS_PARENT_MAP.keys():
            total_logprob, selected_logprobs = sequence_logprob(pipe.model.model, pipe.tokenizer.model, sensation, prompt)
            print(f'sensation for image {ID} is {sensation} with score {(total_logprob, selected_logprobs)}')
            with open(os.path.join(args.result_path, 'results', args.project_name, f'sensations_predicted_by_{args.LLM}.csv'), 'a') as f:
                writer = csv.writer(f)
                writer.writerow([ID, description, sensation, total_logprob, selected_logprobs])

