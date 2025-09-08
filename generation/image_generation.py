import pandas as pd
from configs.inference_config import get_args
from model.pipeline import AdvertisementImageGeneration
from Evaluation.metrics import Metrics
import json
import os
from datetime import datetime
import csv
from utils.data.trian_test_split import get_test_data
import random


def get_prompt_info(args):
    QA = json.load(open(os.path.join(args.data_path, args.test_set_QA)))
    sensations = json.load(open(os.path.join(args.data_path, args.test_set_sensations)))
    return QA, sensations


def save_image(args, filename, image, experiment_datetime, sensation):
    subdirectory = filename.split('/')[0]
    if args.text_input_type == 'AR':
        text_input = 'AR'
    elif args.text_input_type == 'LLM':
        text_input = '_'.join([args.LLM, 'generated_prompt'])
    else:
        text_input = args.description_file.split('/')[-1].split('.')[0]
    directory = os.path.join(args.result_path,
                             'generated_images',
                             args.project_name,
                             experiment_datetime,
                             '_'.join([text_input, args.AD_type, args.T2I_model]),
                             sensation,
                             subdirectory)
    if not os.path.exists(directory):
        os.makedirs(directory)

    image.save(os.path.join(args.result_path,
                            'generated_images',
                            args.project_name,
                            experiment_datetime,
                            '_'.join([text_input, args.AD_type, args.T2I_model]),
                            sensation,
                            filename))


def save_results(args, prompt, action_reason, filename, experiment_datetime, sensation):
    if args.text_input_type == 'AR':
        text_input = 'AR'
    elif args.text_input_type == 'LLM':
        text_input = '_'.join([args.LLM, 'generated_prompt'])
    else:
        text_input = args.description_file.split('/')[-1].split('.')[0]
    directory = os.path.join(args.result_path, 'results', args.project_name)
    if not os.path.exists(directory):
        os.makedirs(directory)

    csv_file_name = '_'.join([text_input, args.AD_type, args.T2I_model, experiment_datetime])
    csv_file_name = f'{csv_file_name}.csv'
    csv_file = os.path.join(directory, csv_file_name)
    if not os.path.exists(csv_file):
        with open(csv_file, 'w', newline='') as file:
            writer = csv.writer(file)

            # Write the header
            writer.writerow(['image_url',
                             'action_reason',
                             'T2I_prompt',
                             'generated_image_url',
                             'sensation'])
    generated_image_url = os.path.join(args.result_path,
                                       'generated_images',
                                       args.project_name,
                                       experiment_datetime,
                                       '_'.join([text_input, args.AD_type, args.T2I_model]),
                                       sensation,
                                       filename)
    with open(csv_file, 'a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow([filename, action_reason, prompt, generated_image_url, sensation])



def process_action_reason(action_reasons):
    return '\n'.join([f'({i}) {statement}' for i, statement in enumerate(action_reasons)])


def generate_images(args):
    test_set = get_test_data(args)['ID'].values[:290]
    AdImageGeneration = AdvertisementImageGeneration(args)
    QA, sensations = get_prompt_info(args)
    experiment_datetime = datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f'experiment started at {experiment_datetime}')
    test_set_image_url = list(test_set)
    test_set_image_url = test_set_image_url
    if args.text_input_type == 'original_description':
        test_set_image_url = pd.read_csv(args.description_file).ID.values
    for filename, content in QA.items():
        if filename not in test_set_image_url:
            continue
        
        action_reasons = content[0]
        image_sensations = sensations[filename]
        for sensation in image_sensations:
            image, prompt = AdImageGeneration(filename, sensation)
            save_image(args, filename, image, experiment_datetime, sensation)
            save_results(args, prompt, action_reasons, filename, experiment_datetime, sensation)
            print(f'image url: {filename}')
            print(f'sensation: {sensation}')
            print(f'action-reason statements: {process_action_reason(action_reasons)}')
            print('-' * 20)
        
    finish_datetime = datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f'experiment ended at {finish_datetime}')


if __name__ == '__main__':
    args = get_args()
    generate_images(args)
