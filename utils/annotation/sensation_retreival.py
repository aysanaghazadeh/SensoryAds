from MLLMs.MLLM import MLLM
from utils.data.physical_sensations import SENSATION_HIERARCHY
from utils.prompt_engineering.prompt_generation import generate_prompt
import os
from PIL import Image
import json

def get_model(
        args
    ):
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
        response = model(image, prompt)
    else:
        response = model(prompt)
    answer_index = ''.join(i for i in response if i.isdigit())
    answer = sensations[answer_index]
    return answer

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
    sensation = retreive_single_sensation(args, model, image, prompt)
    if isinstance(sensations, dict):
        return sensation + ',',  retreive_sensation(args, model, image, sensations[sensation])
    else:
        return sensation

def process_images(
        args, 
        image_list:list
    ):
    
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
        image_sensation_map[image_url] = retreive_sensation(args, model, image, sensations)
        if results_path:
            with open(results_path, 'w') as f:
                json.dump(image_sensation_map, f) 
    return image_sensation_map
