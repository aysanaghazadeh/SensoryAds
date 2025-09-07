from MLLMs.MLLM import MLLM
from LLMs.LLM import LLM
from utils.data.physical_sensations import SENSATION_HIERARCHY
from utils.prompt_engineering.prompt_generation import generate_prompt
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

def retreive_single_sensation(
        args, 
        model, 
        image, 
        prompt, 
        sensations
    ):
    assert args.model_type == 'LLM' or image is not None, 'No image is provided for MLLM'
    if image:
        responses = model(image, prompt)
    else:
        responses = model(prompt)
    responses = responses.split(',')
    answer_indices = [int(''.join(i for i in response if i.isdigit())) for response in responses]
    answers = [sensations[answer_index] for answer_index in answer_indices]
    return answer

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
        image, 
        sensations
    ):
    sensations_list = get_child_sensations(sensations)
    options = get_options(sensations_list)
    data = {
        'options': options
    }
    prompt = generate_prompt(args, data)
    sensation = retreive_single_sensation(args, model, image, prompt, sensations_list)
    if isinstance(sensations, dict):
        return sensation + ',' +  retreive_sensation(args, model, image, sensations[sensation])
    else:
        data = {'physical_sensation': sensation}
        prompt_file = 'extract_visual_elements.jinja'
        MLLM_prompt = args.MLLM_prompt
        args.MLLM_prompt = prompt_file
        prompt = generate_prompt(args, data)
        visual_elements = f'Visual elements:{retrieve_visual_elements(model, image, prompt)}'
        args.MLLM_prompt = MLLM_prompt
        sensation = sensation + '\n' + visual_elements
        return sensation

def process_images(
        args, 
        image_list:list
    ):
    print(f'processing {len(image_list)} stated ...')
    print('-' * 100)
    
    model = get_model(args)
    results_path = os.path.join(args.result_path, args.result_filename) if args.result_filename else None
    if results_path is not None and os.path.exists(results_path) and args.resume:
        image_sensation_map = json.load(open(results_path))
    else:
        image_sensation_map = {}
    for image_url in image_list:
        if image_url in image_sensation_map:
            continue
        image_path = os.path.join(args.data_path, args.test_set_images, image_url)
        image = Image.open(image_path)
        sensations = SENSATION_HIERARCHY
        image_sensation_info = retreive_sensation(args, model, image, sensations)
        image_sensation_map[image_url] = {}
        image_sensation_map[image_url]['sensation'] = image_sensation_info.split('Visual elements:')[0].split(',')
        image_sensation_map[image_url]['visual_elements'] = image_sensation_info.split('Visual elements:')[-1].split(',')
        print(f'sensation info for image {image_url} is: \n {json.dumps(image_sensation_map[image_url], indent=4)}')
        print('-' * 100)
        if results_path:
            with open(results_path, 'w') as f:
                json.dump(image_sensation_map, f) 
    return image_sensation_map
