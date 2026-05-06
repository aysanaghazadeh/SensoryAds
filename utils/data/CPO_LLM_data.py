import random
from accelerate import PartialState
from utils.data.trian_test_split import get_train_data
from datasets import Dataset
import pandas as pd
import os
import json
from transformers import AutoTokenizer
from utils.data.physical_sensations import SENSATIONS_PARENT_MAP


def load_cpo_tokenizer(args):
    """Tokenizer only — avoids loading a second full LLM (and breaking accelerate device_map)."""
    tok = os.environ.get("HF_TOKEN")
    specs = {
        "LLAMA3_instruct": ("meta-llama/Meta-Llama-3-8B-Instruct", {"token": tok, "trust_remote_code": True}),
        "LLAMA3": ("meta-llama/Meta-Llama-3-8B", {"token": tok}),
        "Mistral7B": ("mistralai/Mistral-7B-v0.1", {}),
        "Mistral7BInstruct": ("mistralai/Mistral-7B-Instruct-v0.2", {}),
        "phi": ("microsoft/Phi-3-mini-4k-instruct", {"token": tok, "trust_remote_code": True}),
        "vicuna": ("lmsys/vicuna-13b-v1.5", {"token": tok}),
        "InternLM": ("internlm/internlm3-8b-instruct", {"trust_remote_code": True}),
        "QWenLM": ("Qwen/Qwen2.5-7B-Instruct", {"token": tok, "trust_remote_code": True}),
    }
    if args.LLM not in specs:
        raise ValueError(
            f"load_cpo_tokenizer: no tokenizer spec for LLM={args.LLM!r}; pass tokenizer= to get_LLM_CPO_training_data."
        )
    model_id, kw = specs[args.LLM]
    return AutoTokenizer.from_pretrained(model_id, **kw)


def get_LLM_CPO_training_data(args, image_urls, tokenizer=None):
    if tokenizer is None:
        tokenizer = load_cpo_tokenizer(args)
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


def get_train_LLM_CPO_Dataloader(args, tokenizer=None):
    image_urls = get_train_data(args)
    dataset = get_LLM_CPO_training_data(args, image_urls, tokenizer=tokenizer)
    return dataset
