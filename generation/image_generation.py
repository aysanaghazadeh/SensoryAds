import pandas as pd
from configs.inference_config import get_args
from model.pipeline import AdvertisementImageGeneration
from Evaluation.AD_metrics import Metrics
import json
import os
from datetime import datetime
import csv
from utils.data.trian_test_split import get_test_data
from utils.data.physical_sensations import SENSATION_OPPOSITES
import random
from PIL import Image

SENSATION_OPPOSITES_LOWER = {k.lower(): v for k, v in SENSATION_OPPOSITES.items()}


def get_opposite_sensation(sensation):
    """Case-insensitive lookup into SENSATION_OPPOSITES. Returns None if the
    sensation has no entry (or no opposite) in the map."""
    return SENSATION_OPPOSITES_LOWER.get(sensation.lower())


def get_prompt_info(args):
    QA = json.load(open(os.path.join(args.data_path, args.test_set_QA)))
    sensations = {}
    if args.with_physical_sensation:
        sensations = json.load(open(os.path.join(args.data_path, args.test_set_sensation)))
    return QA, sensations


def save_image(args, filename, image, experiment_datetime, sensation):
    subdirectory = filename.split('/')[0]
    if args.text_input_type == 'AR':
        text_input = 'AR'
    elif args.text_input_type == 'LLM':
        text_input = '_'.join([args.LLM, 'generated_prompt'])
    else:
        text_input = args.description_file.split('/')[-1].split('.')[0]
    if args.with_physical_sensation:
        directory = os.path.join(args.result_path,
                                 'generated_images',
                                 args.project_name,
                                 experiment_datetime,
                                 '_'.join([text_input, 'ALL', args.T2I_model]),
                                 sensation,
                                 subdirectory)
    else:
        directory = os.path.join(args.result_path,
                                 'generated_images',
                                 args.project_name,
                                 experiment_datetime,
                                 '_'.join([text_input, 'ALL', args.T2I_model]),
                                 subdirectory)
    if not os.path.exists(directory):
        os.makedirs(directory)

    # Reuses `directory` as-is (rather than rebuilding the path) so this can
    # never drift from the with_physical_sensation branch above again.
    image.save(os.path.join(directory, os.path.basename(filename)))


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

    csv_file_name = '_'.join([text_input, 'ALL', args.T2I_model, experiment_datetime])
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
    if args.with_physical_sensation:
        generated_image_url = os.path.join(args.result_path,
                                           'generated_images',
                                           args.project_name,
                                           experiment_datetime,
                                           '_'.join([text_input, 'ALL', args.T2I_model]),
                                           sensation,
                                           filename)
    else:
        generated_image_url = os.path.join(args.result_path,
                                           'generated_images',
                                           args.project_name,
                                           experiment_datetime,
                                           '_'.join([text_input, 'ALL', args.T2I_model]),
                                           filename)
    with open(csv_file, 'a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow([filename, action_reason, prompt, generated_image_url, sensation])



def process_action_reason(action_reasons):
    return '\n'.join([f'({i}) {statement}' for i, statement in enumerate(action_reasons)])


def generate_images(args):
    test_set = get_test_data(args)
    AdImageGeneration = AdvertisementImageGeneration(args)
    QA, sensations = get_prompt_info(args)
    experiment_datetime = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.experiment_datetime:
        experiment_datetime = args.experiment_datetime
    
    print(f'experiment started at {experiment_datetime}')
    test_set_image_url = list(test_set)
    test_set_image_url = test_set_image_url
    if args.text_input_type == 'original_description':
        test_set_image_url = pd.read_csv(args.description_file).ID.values
    for filename, content in QA.items():
        if filename not in test_set_image_url:
            continue
        if args.with_physical_sensation and filename not in sensations:
            continue

        action_reasons = content[0]
        if args.with_physical_sensation:
            image_sensations = sensations[filename]['image_sensations']
            if args.find_sensation:
                if len(image_sensations) > 1:
                    image_sensations = [image_sensations[0]]
        else:
            # Single pass, no sensation involved: 'no sensation' round-trips
            # cleanly through the existing sensation.replace(' sensation', '')
            # cALLs below into a grammatical "no sensation" prompt/folder name.
            image_sensations = ['no sensation']
        for sensation in image_sensations:
            target_sensation = sensation
            if args.use_opposite_sensation:
                opposite_sensation = get_opposite_sensation(sensation)
                if not opposite_sensation:
                    print(f'sensation {sensation} has no opposite in SENSATION_OPPOSITES and will be skipped...')
                    continue
                target_sensation = opposite_sensation.lower()
            if args.experiment_datetime:
                run_dir = f'../experiments/generated_images/SensoryAds/{args.experiment_datetime}/{args.text_input_type}_ALL_{args.T2I_model}'
                if args.with_physical_sensation:
                    image_path = os.path.join(run_dir, target_sensation, filename)
                else:
                    image_path = os.path.join(run_dir, filename)
                if os.path.exists(image_path):
                    print(f'image {filename} for sensation {target_sensation} already exists and will be skipped...')
                    continue
            if args.T2I_model == 'AgenticEditing':
                if args.Editing_model == 'FluxKontext':
                    generated_image = Image.open(os.path.join('../experiments/generated_images/SensoryAds/20250916_122348/AR_ALL_Flux', sensation, filename))
                elif args.Editing_model == 'QwenImageEdit':
                    generated_image = Image.open(os.path.join('../experiments/generated_images/SensoryAds/20260224_024108/AR_ALL_QWenImage', sensation, filename))
                    # generated_image = Image.new("RGB", (1024, 1024), (255, 255, 255))
                elif args.Editing_model == 'SD3ControlnetEdit':
                    generated_image = Image.open(os.path.join('../experiments/generated_images/SensoryAds/20251123_225258/AR_ALL_SD3', sensation, filename)) # SD3-Controlnet 20260311_093216
                else:
                    raise ValueError(f'Editing model {args.Editing_model} not supported')
                image, prompt = AdImageGeneration(image_filename=filename, sensation=target_sensation.replace(' sensation', ''), generated_image=generated_image, prompt=process_action_reason(action_reasons))
            else:
                image, prompt = AdImageGeneration(image_filename=filename, sensation=target_sensation.replace(' sensation', ''))
            if image is None:
                continue
            save_image(args, filename, image, experiment_datetime, target_sensation)
            save_results(args, prompt, action_reasons, filename, experiment_datetime, target_sensation)
            print(f'image url: {filename}')
            print(f'sensation: {target_sensation}')
            print(f'action-reason statements: {process_action_reason(action_reasons)}')
            print('-' * 20)
        
    finish_datetime = datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f'experiment ended at {finish_datetime}')


if __name__ == '__main__':
    args = get_args()
    generate_images(args)
