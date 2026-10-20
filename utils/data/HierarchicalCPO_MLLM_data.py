from utils.data.trian_test_split import get_train_data
from datasets import Dataset
import os
import json
from utils.data.physical_sensations import SENSATIONS_PARENT_MAP

PROMPT = "Sensation that the image evokes the most is: "


def get_MLLM_HierarchicalCPO_training_data(args, image_urls):
    LOWER_SENSATIONS_PARENT_MAP = {}
    for sensation in SENSATIONS_PARENT_MAP:
        LOWER_SENSATIONS_PARENT_MAP[sensation.lower()] = SENSATIONS_PARENT_MAP[sensation].lower()
    # Rows only hold the image path and the three completions; images are
    # opened and tokenized per batch by the collator, since keeping every
    # pair's image in memory (as DPO_MLLM_data does) would not fit.
    dataset = {'image_url': [], 'prompt': [], 'chosen': [], 'rejected': [], 'parent_of_chosen': []}
    sensations = json.load(open(os.path.join(args.data_path, args.sensation_annotations)))
    for image_url in image_urls:
        if image_url not in sensations:
            continue
        sensation_scores = sensations[image_url]['sensation_scores']
        # Unordered pairs only: (a, b) and (b, a) resolve to the same chosen/rejected.
        sensation_list = [sensation.strip() for sensation in sensation_scores]
        for i, sensation1 in enumerate(sensation_list):
            for sensation2 in sensation_list[i + 1:]:
                if sensation_scores[sensation1] == sensation_scores[sensation2]:
                    continue
                if sensation_scores[sensation1] > sensation_scores[sensation2]:
                    chosen_answer, rejected_answer = sensation1, sensation2
                else:
                    chosen_answer, rejected_answer = sensation2, sensation1
                dataset['image_url'].append(image_url)
                dataset['prompt'].append(PROMPT)
                dataset['chosen'].append(chosen_answer)
                dataset['rejected'].append(rejected_answer)
                dataset['parent_of_chosen'].append(LOWER_SENSATIONS_PARENT_MAP[chosen_answer])
    print(f'total number of sensation pairs: {len(dataset["prompt"])}')
    return Dataset.from_dict(dataset)


def get_train_MLLM_HierarchicalCPO_Dataloader(args):
    image_urls = get_train_data(args)
    return get_MLLM_HierarchicalCPO_training_data(args, image_urls)
