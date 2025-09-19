import random
from accelerate import PartialState
from utils.data.trian_test_split import get_train_data
from datasets import Dataset
import pandas as pd
import os
import json
from LLMs.LLM import LLM
from utils.data.physical_sensations import SENSATIONS_PARENT_MAP


# def get_tokenizer(args):
#     pipe = LLM(args)
#     tokenizer = pipe.model.tokenizer
#     # pipe.model.model = pipe.model.model.to(device='cpu')
#     return tokenizer


def get_LLM_HierarchicalCPO_training_data(args, tokenizer, image_urls):
    tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.chat_template is None:
        tokenizer.chat_template = "{% if not add_generation_prompt is defined %}{% set add_generation_prompt = false %}{% endif %}{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"

    def process(row):
        # Important: The CPODataCollator expects prompts to be part of the chosen/rejected sequences
        prompt = row['prompt'] + "\n\n"

        # Tokenize all four fields
        tokenized_chosen = tokenizer(prompt + row['chosen'], truncation=True)
        tokenized_rejected = tokenizer(prompt + row['rejected'], truncation=True)
        tokenized_parent = tokenizer(prompt + row['parent_of_chosen'], truncation=True)

        return {
            "prompt": prompt,
            "chosen_input_ids": tokenized_chosen["input_ids"],
            "chosen_attention_mask": tokenized_chosen["attention_mask"],
            "rejected_input_ids": tokenized_rejected["input_ids"],
            "rejected_attention_mask": tokenized_rejected["attention_mask"],
            "parent_of_chosen_input_ids": tokenized_parent["input_ids"],
            "parent_of_chosen_attention_mask": tokenized_parent["attention_mask"],
        }

    descriptions = pd.read_csv(args.description_file)
    dataset = {'prompt': [], 'chosen': [], 'rejected': []}
    sensations = json.load(open(os.path.join(args.data_path, args.sensation_annotations)))
    for image_url in image_urls:
        if image_url in sensations:
            sensation_scores = sensations[image_url]['sensation_scores']
            description = descriptions.loc[descriptions['ID'] == image_url]['description'].values[0].split('Q2:')[-1]
            prompt = f"""Context: Description of an image is {description}.
                         Sensation that the image evokes the most is: """
            for sensation1 in sensation_scores:
                for sensation2 in sensation_scores:
                    sensation1 = sensation1.strip()
                    sensation2 = sensation2.strip()
                    if sensation_scores[sensation1] == sensation_scores[sensation2]:
                        continue

                    chosen_answer = sensation1 if sensation_scores[sensation1] > sensation_scores[sensation2] else sensation2
                    rejected_answer = sensation1 if sensation_scores[sensation1] < sensation_scores[sensation2] else sensation2

                    chosen = [{'content': prompt, 'role': 'user'},
                              {'content': chosen_answer, 'role': 'assistant'}]
                    parent_of_chosen = [{'content': prompt, 'role': 'user'},
                              {'content': SENSATIONS_PARENT_MAP[chosen_answer], 'role': 'assistant'}]
                    rejected = [{'content': prompt, 'role': 'user'},
                                {'content': rejected_answer, 'role': 'assistant'}]
                    dataset['prompt'].append(prompt)
                    dataset['chosen'].append(chosen)
                    dataset['rejected'].append(rejected)
                    dataset['parent_of_chosen'].append(parent_of_chosen)
    dataset = Dataset.from_dict(dataset)
    with PartialState().local_main_process_first():
        ds = dataset.map(process)

    train_dataset = ds
    return train_dataset


def get_train_LLM_HierarchicalCPO_Dataloader(args, tokenizer):
    image_urls = get_train_data(args)
    dataset = get_LLM_HierarchicalCPO_training_data(args, tokenizer, image_urls)
    return dataset
