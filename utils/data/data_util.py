import random
from accelerate import PartialState
from utils.data.trian_test_split import get_train_data
from transformers import AutoTokenizer
from datasets import Dataset
import pandas as pd
import os
import json


def get_LLAMA3_CPO_training_data(args, image_urls):
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct",
                                              token=os.environ.get('HF_TOKEN'),
                                              trust_remote_code=True,
                                              padding='right')
    tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.chat_template is None:
        tokenizer.chat_template = "{% if not add_generation_prompt is defined %}{% set add_generation_prompt = false %}{% endif %}{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"

    def process(row):
        row["chosen"] = tokenizer.apply_chat_template(row["chosen"], tokenize=False)
        row["rejected"] = tokenizer.apply_chat_template(row["rejected"], tokenize=False)
        return row

    descriptions = pd.read_csv(args.description_file)
    dataset = {'prompt': [], 'chosen': [], 'rejected': []}
    QAs = json.load(open(os.path.join(args.data_path, args.test_set_QA)))
    negative_QAs = {}
    negative_QAs['reason'] = json.load(
        open(os.path.join(args.data_path, 'train/reason_hard_QA_Combined_Action_Reason_train.json')))
    negative_QAs['action'] = json.load(
        open(os.path.join(args.data_path, 'train/action_hard_QA_Combined_Action_Reason_train.json')))
    negative_QAs['adjective'] = json.load(
        open(os.path.join(args.data_path, 'train/adjective_hard_QA_Combined_Action_Reason_train.json')))
    negative_QAs['semantic'] = json.load(
        open(os.path.join(args.data_path, 'train/semantic_hard_QA_Combined_Action_Reason_train.json')))

    for image_url in image_urls:
        if image_url in negative_QAs['reason']:
            QA = QAs[image_url]
            description = descriptions.loc[descriptions['ID'] == image_url]['description'].values
            prompt = f"""What is the correct interpretation for the described image:
                                         Description: {description}"""
            for AR in QA[0]:
                for negative_type in negative_QAs:
                    for negative_option in negative_QAs[negative_type][image_url][1]:
                        if (negative_option in negative_QAs[negative_type][image_url][0]) or (negative_option in dataset['rejected']):
                            continue
                        chosen = [{'content': prompt, 'role': 'user'},
                                  {'content': AR, 'role': 'assistant'}]
                        rejected = [{'content': prompt,
                                     'role': 'user'},
                                    {'content': negative_option,
                                     'role': 'assistant'}]
                        dataset['prompt'].append(prompt)
                        dataset['chosen'].append(chosen)
                        dataset['rejected'].append(rejected)
    dataset = Dataset.from_dict(dataset)
    with PartialState().local_main_process_first():
        ds = dataset.map(process)

    train_dataset = ds
    return train_dataset


def get_train_LLAMA3_CPO_Dataloader(args):
    image_urls = list(json.load(open(os.path.join(args.data_path,
                                                  'train/reason_hard_QA_Combined_Action_Reason_train.json'))).keys())
    dataset = get_LLAMA3_CPO_training_data(args, image_urls)
    return dataset
