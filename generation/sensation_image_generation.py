from utils.data.physical_sensations import SENSATIONS_PARENT_MAP
from model.pipeline import AdvertisementImageGeneration
import os
from datetime import datetime
import csv


def save_image(args, filename, image, experiment_datetime, sensation):
    directory = os.path.join(args.result_path,
                           'generated_images',
                           args.project_name,
                           experiment_datetime,
                           args.AD_type,
                           sensation)
    if not os.path.exists(directory):
        os.makedirs(directory)

    image.save(os.path.join(args.result_path,
                           'generated_images',
                           args.project_name,
                           experiment_datetime,
                           args.AD_type,
                           sensation,
                           filename))


def save_results(args, prompt, filename, experiment_datetime, sensation):
    directory = os.path.join(args.result_path, 'results', args.project_name)
    if not os.path.exists(directory):
        os.makedirs(directory)

    csv_file_name = '_'.join([args.AD_type, args.T2I_model, experiment_datetime])
    csv_file_name = f'{csv_file_name}.csv'
    csv_file = os.path.join(directory, csv_file_name)
    if not os.path.exists(csv_file):
        with open(csv_file, 'w', newline='') as file:
            writer = csv.writer(file)

            # Write the header
            writer.writerow(['image_url',
                             'T2I_prompt',
                             'generated_image_url',
                             'sensation'])
    generated_image_url = os.path.join(args.result_path,
                                       'generated_images',
                                       args.project_name,
                                       experiment_datetime,
                                       args.AD_type,
                                       sensation,
                                       filename)
    with open(csv_file, 'a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow([os.path.join(sensation, filename), prompt, generated_image_url, sensation])


def process_action_reason(action_reasons):
    return '\n'.join([f'({i}) {statement}' for i, statement in enumerate(action_reasons)])


def generate_images(args):
    sensation_list = list(SENSATIONS_PARENT_MAP.keys())
    AdImageGeneration = AdvertisementImageGeneration(args)
    experiment_datetime = datetime.now().strftime("%Y%m%d_%H%M%S")
    for sensation in sensation_list:
        for i in range(10):
            filename = f'{i}.png'
            image, prompt = AdImageGeneration(filename, sensation.replace(' sensation', ''), seed=i)
            save_image(args, filename, image, experiment_datetime, sensation)
            save_results(args, prompt, filename, experiment_datetime, sensation)
            print(f'image url: {sensation}/{filename}')
            print(f'sensation: {sensation}')
            print('-' * 20)

    finish_datetime = datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f'experiment ended at {finish_datetime}')

