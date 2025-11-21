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
        # 1. Apply the chat template to format the message lists into single strings
        chosen_str = tokenizer.apply_chat_template(row["chosen"], tokenize=False)
        rejected_str = tokenizer.apply_chat_template(row["rejected"], tokenize=False)
        # Don't forget the parent for your hierarchical loss!
        parent_str = tokenizer.apply_chat_template(row["parent_of_chosen"], tokenize=False)

        # 2. Tokenize the formatted strings to get the required input_ids and attention_masks
        tokenized_chosen = tokenizer(chosen_str)
        tokenized_rejected = tokenizer(rejected_str)
        tokenized_parent = tokenizer(parent_str)
        return {
            "prompt": "",
            "chosen_input_ids": tokenized_chosen["input_ids"],
            "chosen_attention_mask": tokenized_chosen["attention_mask"],
            "rejected_input_ids": tokenized_rejected["input_ids"],
            "rejected_attention_mask": tokenized_rejected["attention_mask"],
            "parent_of_chosen_input_ids": tokenized_parent["input_ids"],
            "parent_of_chosen_attention_mask": tokenized_parent["attention_mask"],
        }

    LOWER_SENSATIONS_PARENT_MAP = {}
    for sensation in SENSATIONS_PARENT_MAP:
        LOWER_SENSATIONS_PARENT_MAP[sensation.lower()] = SENSATIONS_PARENT_MAP[sensation].lower()
    descriptions = pd.read_csv(args.description_file)
    dataset = {'prompt': [], 'chosen': [], 'rejected': [], 'parent_of_chosen': []}
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
                              {'content': LOWER_SENSATIONS_PARENT_MAP[chosen_answer], 'role': 'assistant'}]
                    rejected = [{'content': prompt, 'role': 'user'},
                                {'content': rejected_answer, 'role': 'assistant'}]
                    dataset['prompt'].append(prompt)
                    dataset['chosen'].append(chosen)
                    dataset['rejected'].append(rejected)
                    dataset['parent_of_chosen'].append(parent_of_chosen)
    print(f'total number of sensation pairs: {len(dataset["prompt"])}')
    dataset = Dataset.from_dict(dataset)
    with PartialState().local_main_process_first():
        ds = dataset.map(process, batched=False)

    train_dataset = ds
    return train_dataset


def get_train_LLM_HierarchicalCPO_Dataloader(args, tokenizer):
    image_urls = get_train_data(args)
    dataset = get_LLM_HierarchicalCPO_training_data(args, tokenizer, image_urls)
    return dataset
