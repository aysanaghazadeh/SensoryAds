from MLLMs.MLLM import MLLM
from LLMs.LLM import LLM
from utils.data.physical_sensations import SENSATION_HIERARCHY, SENSATION_DEFINITION
from utils.prompt_engineering.prompt_generation import generate_text_generation_prompt
import os
from PIL import Image
import json


def get_model(
        args
    ):
    if args.model_type == 'LLM':
        model = LLM(args)
    if args.model_type == 'MLLM':
        model = MLLM(args)
    return model

def retreive_single_level_sensation(
        args, 
        model, 
        prompt, 
        sensations,
        image=None,
    ):
    assert args.model_type == 'LLM' or image is not None, 'No image is provided for MLLM'
    if image:
        responses = model(image, prompt)
    else:
        responses = model(prompt)
    responses = responses.split(',')
    answer_indices = [int(''.join(i for i in response if i.isdigit())) for response in responses]
    answers = [sensations[answer_index] for answer_index in answer_indices]
    return answers

def retrieve_visual_elements(
        model,
        image,
        prompt
    ):
    response = model(image, prompt)
    return response

def get_child_sensations(
        sensations
    ):
    if isinstance(sensations, list):
        return sensations
    elif isinstance(sensations, dict):
        return list(sensations.keys())
    else:
        return []

def get_options(
        sensations:list
    ):
    options = '\n'.join([f'{i}- {sensation}' for i, sensation in enumerate(sensations)])
    return options

def retreive_sensation(
        args, 
        model, 
        sensations_map,
        parent_sensation='root',
        description=None,
        image=None, 
    ):
    sensations_list = get_child_sensations(sensations_map)
    options = get_options(sensations_list)
    data = {
        'options': options,
        'context': SENSATION_DEFINITION[parent_sensation],
        'description': description
    }
    prompt = generate_text_generation_prompt(args, data)
    sensations= retreive_single_level_sensation(args, model, image, prompt, sensations_list)
    if isinstance(sensations_map, dict):
        output_list = []
        for sensation in sensations:
            if sensation is not None:
                output_list += [sensation + ',' + retrieved_sensation in retrieve_sensation(arge, 
                                                                                            model, 
                                                                                            image, 
                                                                                            sensations_map[sensation], 
                                                                                            parent_sensation=sensation)]
        return output_list
    else:
        return sensations

def process_files(
        args, 
        image_list:list
    ):
    print(f'processing {len(image_list)} file started ...')
    print('-' * 100)
    assert args.model_type == 'MLLM' or args.description_file is not None, 'description file is missing'
    if args.model_type == 'LLM':
        descriptions = pd.read_csv(args.description_file)
    model = get_model(args)
    results_path = os.path.join(args.result_path, 
                                args.result_filename) if args.result_filename else os.path.join(
                                args.result_path,
                                'results',
                                args.project_name,
                                '_'.join([
                                        args.inference_type,
                                        args.task,
                                        args.AD_type,
                                        args.MLLM if args.model_type == 'MLLM' else args.LLM,
                                        args.MLLM_prompt if args.model_type == 'MLLM' else args.LLM_prompt
                                    ]).replace('.jinja', '.json')
                                
                            )
    if results_path is not None and os.path.exists(results_path) and args.resume:
        image_sensation_map = json.load(open(results_path))
    else:
        image_sensation_map = {}
    for image_url in image_list:
        if image_url in image_sensation_map:
            continue
        if args.model_type == 'MLLM':
            image_path = os.path.join(args.data_path, args.test_set_images, image_url)
            image = Image.open(image_path)
        elif args.model_type == "LLM":
            description = descriptions.loc[descriptions['ID'] == image_url]['description'].values[0]
        sensations = SENSATION_HIERARCHY
        if args.model_type == 'MLLM':
            image_sensations = retreive_sensation(args, model, sensations, image=image)
        elif args.model_type == 'LLM':
            image_sensations = retreive_sensation(args, model, description=description)
        image_sensation_map[image_url] = image_sensations
        # image_sensation_map[image_url]['sensation'] = image_sensation_info.split('Visual elements:')[0].split(',')
        # image_sensation_map[image_url]['visual_elements'] = image_sensation_info.split('Visual elements:')[-1].split(',')
        print(f'sensation info for image {image_url} is: \n {json.dumps(image_sensation_map[image_url], indent=4)}')
        print('-' * 100)
        if results_path:
            with open(results_path, 'w') as f:
                json.dump(image_sensation_map, f) 
    return image_sensation_map
