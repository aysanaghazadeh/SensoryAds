from accelerate import PartialState
from utils.data.trian_test_split import get_train_data
from utils.data.CPO_LLM_data import load_cpo_tokenizer
from datasets import Dataset
import pandas as pd
import os
import json


def get_LLM_SFT_score_training_data(args, image_urls, tokenizer=None):
    if tokenizer is None:
        tokenizer = load_cpo_tokenizer(args)
    tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.chat_template is None:
        tokenizer.chat_template = "{% if not add_generation_prompt is defined %}{% set add_generation_prompt = false %}{% endif %}{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"

    def process(row):
        row["text"] = tokenizer.apply_chat_template(row["messages"], tokenize=False)
        return row

    descriptions = pd.read_csv(args.description_file)
    description_by_id = dict(zip(descriptions['ID'], descriptions['description']))
    dataset = {'messages': []}
    sensations = json.load(open(os.path.join(args.data_path, args.sensation_annotations)))
    for image_url in image_urls:
        if image_url in sensations:
            # A handful of IDs carry stray whitespace from the original
            # annotation parse and don't match description_file's clean copy.
            if image_url not in description_by_id:
                continue
            sensation_scores = sensations[image_url]['sensation_scores']
            description = description_by_id[image_url].split('Q2:')[-1]
            for sensation, score in sensation_scores.items():
                sensation = sensation.strip()
                prompt = f"""Image Description: {description}
Question: What is the {sensation} sensation intensity of the described image?"""
                dataset['messages'].append([{'content': prompt, 'role': 'user'},
                                            {'content': str(score), 'role': 'assistant'}])
    dataset = Dataset.from_dict(dataset)
    with PartialState().local_main_process_first():
        ds = dataset.map(process, remove_columns=['messages'])
    return ds


def get_train_LLM_SFT_score_Dataloader(args, tokenizer=None):
    image_urls = get_train_data(args)
    dataset = get_LLM_SFT_score_training_data(args, image_urls, tokenizer=tokenizer)
    return dataset
