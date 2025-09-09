import random
from accelerate import PartialState
from utils.data.trian_test_split import get_train_data
from datasets import Dataset
import pandas as pd
import os
import json
from LLMs.LLM import LLM


def get_tokenizer(args):
    pipe = LLM(args)
    tokenizer = pipe.model.tokenizer
    return tokenizer


def get_LLAMA3_CPO_training_data(args, image_urls):
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
    for image_url in image_urls:
        if image_url in sensations:
            sensation_scores = sensations[image_url]['sensation_scores']
            description = descriptions.loc[descriptions['ID'] == image_url]['description'].values
            prompt = f"""Context: Description of an image is {description}
            Given the description of the image, the sensation that the image evokes is: """
            for sensation1 in sensation_scores:
                for sensation2 in sensation_scores:
                    if sensation_scores[sensation1] == sensation_scores[sensation2]:
                        continue
                    chosen_answer = sensation1 if sensation_scores[sensation1] > sensation_scores[sensation2] else sensation2
                    rejected_answer = sensation1 if sensation_scores[sensation1] < sensation_scores[sensation2] else sensation2

                    chosen = [{'content': prompt, 'role': 'user'},
                              {'content': chosen_answer, 'role': 'assistant'}]
                    rejected = [{'content': prompt, 'role': 'user'},
                                {'content': rejected_answer, 'role': 'assistant'}]
                    dataset['prompt'].append(prompt)
                    dataset['chosen'].append(chosen)
                    dataset['rejected'].append(rejected)
    dataset = Dataset.from_dict(dataset)
    with PartialState().local_main_process_first():
        ds = dataset.map(process)

    train_dataset = ds
    return train_dataset


def get_train_LLAMA3_CPO_Dataloader(args):
    image_urls = get_train_data(args)
    dataset = get_LLAMA3_CPO_training_data(args, image_urls)
    return dataset
