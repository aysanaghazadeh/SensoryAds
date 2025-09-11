import random
from accelerate import PartialState
from utils.data.trian_test_split import get_train_data
from datasets import Dataset
import pandas as pd
import os
import json
from utils.data.physical_sensations import SENSATIONS_PARENT_MAP
from PIL import Image

def get_processor(pipe):
    processor = pipe.model.processor
    return processor


def get_MLLM_DPO_training_data(args, image_urls):

    dataset = {'question': [], 'chosen': [], 'rejected': [], 'image': []}
    sensations = json.load(open(os.path.join(args.data_path, args.sensation_annotations)))
    options = '-'.join([f'{i}. {option}' for i, option in enumerate(SENSATIONS_PARENT_MAP.keys())])
    sensation_index_map = {}
    for i, option in enumerate(SENSATIONS_PARENT_MAP.keys()):
        sensation_index_map[option.lower()] = str(i)
    for image_url in image_urls:
        if image_url in sensations:
            image = Image.open(os.path.join(args.data_path, args.test_set_images, image_url))
            sensation_scores = sensations[image_url]['sensation_scores']

            prompt = f"""<image>.
                        options: {options}
                        Given the the image, the sensation evoked by the image the most is:"""
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
                    dataset['question'].append(prompt)
                    dataset['chosen'].append(chosen)
                    dataset['rejected'].append(rejected)
                    dataset['image'].append(image)
    dataset = Dataset.from_dict(dataset)
    return dataset


def get_train_MLLM_DPO_Dataloader(args):
    image_urls = get_train_data(args)
    dataset = get_MLLM_DPO_training_data(args, image_urls)
    return dataset
