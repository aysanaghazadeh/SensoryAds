import random
from accelerate import PartialState
from utils.data.trian_test_split import get_train_data
from datasets import Dataset
import pandas as pd
import os
import json
from LLMs.LLM import LLM
from utils.data.physical_sensations import SENSATIONS_PARENT_MAP


def get_tokenizer(args):
    pipe = LLM(args)
    tokenizer = pipe.model.tokenizer
    pipe.model.model = pipe.model.model.to(device='cpu')
    return tokenizer


def get_LLM_CPO_training_data(args, image_urls):
    tokenizer = get_tokenizer(args)
    tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.chat_template is None:
        tokenizer.chat_template = "{% if not add_generation_prompt is defined %}{% set add_generation_prompt = false %}{% endif %}{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"

    def process(row):
        row["chosen"] = tokenizer.apply_chat_template(row["chosen"], tokenize=False)
        row["rejected"] = tokenizer.apply_chat_template(row["rejected"], tokenize=False)
        return row

    descriptions = pd.read_csv(args.description_file)
    dataset = {'prompt': [], 'chosen': [], 'rejected': []}
    sensations = json.load(open(os.path.join(args.data_path, args.sensation_annotations)))
    options = '-'.join([f'{i}. {option}' for i, option in enumerate(SENSATIONS_PARENT_MAP.keys())])
    sensation_index_map = {}
    for i, option in enumerate(SENSATIONS_PARENT_MAP.keys()):
        sensation_index_map[option.lower()] = str(i)
    for image_url in image_urls:
        if image_url in sensations:
            sensation_scores = sensations[image_url]['sensation_scores']
            description = descriptions.loc[descriptions['ID'] == image_url]['description'].values[0].split('Q2:')[-1]
            prompt = f"""Context: Description of an image is {description}.
                        options: {options}
                        Given the description of the image, the index of sensation evoked by the image the most is:"""
            for sensation1 in sensation_scores:
                for sensation2 in sensation_scores:
                    sensation1 = sensation1.strip()
                    sensation2 = sensation2.strip()
                    if sensation_scores[sensation1] == sensation_scores[sensation2]:
                        continue

                    chosen_answer = sensation_index_map[sensation1] if sensation_scores[sensation1] > sensation_scores[sensation2] else sensation_index_map[sensation2]
                    rejected_answer = sensation_index_map[sensation1] if sensation_scores[sensation1] < sensation_scores[sensation2] else sensation_index_map[sensation2]

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


def get_train_LLM_CPO_Dataloader(args):
    image_urls = get_train_data(args)
    dataset = get_LLM_CPO_training_data(args, image_urls)
    return dataset
