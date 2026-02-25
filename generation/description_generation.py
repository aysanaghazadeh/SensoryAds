import json
import re
from jinja2 import Environment, FileSystemLoader
from transformers import pipeline
from utils.data.trian_test_split import get_test_data, get_train_data
from PIL import Image
import os
import csv
import pandas as pd
from utils.prompt_engineering.prompt_generation import ImageGenerationPromptGenerator
from LLMs.LLM import LLM
from MLLMs.MLLM import MLLM
from utils.data.physical_sensations import SENSATIONS_PARENT_MAP


_Q1_Q2_RE = re.compile(r'(?is)\bQ1\s*:\s*(.*?)\s*\bQ2\s*:\s*(.*)\s*$')
_LEADING_ENUM_RE = re.compile(r'^\s*\d+\s*[\.\)\-:]\s*')
_DOUBLE_QUOTED_TEXT_RE = re.compile(r'"[^"\n]{1,800}"')
_SINGLE_QUOTED_TEXT_RE = re.compile(r"'[^'\n]{1,800}'")
_I_SHOULD_CLAUSE_RE = re.compile(r"(?i)\bI\s*SHOULD(?:N'T| NOT)?\b[^.!\n\"]{0,800}")
_BECAUSE_CLAUSE_RE = re.compile(r'(?i)\bBECAUSE\b[^.!\n\"]{0,800}')
_READS_SAYS_RE = re.compile(r'(?i)\b(?:reads|says)\b\s*(?:that\s*)?[:,-]?\s*[^.!\n]{1,800}')
_WORD_FOLLOWS_RE = re.compile(r'(?i)\b(?:the\s+word|word)\b\s+([A-Za-z0-9$€£¥]{2,30})')


def _sanitize_q1_list(q1: str) -> str:
    if not isinstance(q1, str):
        return q1
    raw = q1.replace('\n', ' ')
    parts = [p.strip() for p in raw.split(',') if p.strip()]
    cleaned_items = []
    for p in parts:
        item = _LEADING_ENUM_RE.sub('', p).strip()
        if not item:
            continue
        lowered = item.lower()
        if any(k in lowered for k in ['text', 'logo', 'wordmark', 'letters', 'numbers', 'caption', 'slogan']):
            continue
        cleaned_items.append(item)
        if len(cleaned_items) >= 5:
            break
    if not cleaned_items:
        return 'None'
    return ', '.join([f'{i + 1}. {it}' for i, it in enumerate(cleaned_items)])


def _sanitize_q2_paragraph(q2: str) -> str:
    if not isinstance(q2, str):
        return q2
    cleaned = q2
    # Redact obvious OCR / transcription.
    cleaned = _DOUBLE_QUOTED_TEXT_RE.sub('[TEXT]', cleaned)
    cleaned = _SINGLE_QUOTED_TEXT_RE.sub('[TEXT]', cleaned)
    cleaned = _I_SHOULD_CLAUSE_RE.sub('[TEXT]', cleaned)
    cleaned = _BECAUSE_CLAUSE_RE.sub('[TEXT]', cleaned)
    cleaned = _READS_SAYS_RE.sub('reads [TEXT]', cleaned)
    # Replace only the token after "word"/"the word" with [TEXT]
    cleaned = _WORD_FOLLOWS_RE.sub(lambda m: m.group(0).replace(m.group(1), '[TEXT]'), cleaned)
    cleaned = re.sub(r'\s+\[TEXT\]\s+\[TEXT\]\s+', ' [TEXT] ', cleaned)
    cleaned = re.sub(r'\s{2,}', ' ', cleaned).strip()
    return cleaned


def _sanitize_image_descriptor_output(text: str) -> str:
    """
    Best-effort enforcement for prompts that should not include any on-image text.
    Keeps Q1/Q2 format but redacts leaked OCR and removes 'Text/Logo' from Q1.
    """
    if not isinstance(text, str):
        return text
    m = _Q1_Q2_RE.search(text.strip())
    if not m:
        return _sanitize_q2_paragraph(text)
    q1, q2 = m.group(1).strip(), m.group(2).strip()
    q1 = _sanitize_q1_list(q1)
    q2 = _sanitize_q2_paragraph(q2)
    return f'Q1: {q1}\nQ2: {q2}'


def get_model(args):
    # Load model directly
    if args.description_type == 'combine':
        pipe = LLM(args)
        return pipe
    pipe = MLLM(args)
    return pipe


def get_llm(args):
    model = ImageGenerationPromptGenerator(args)
    model.set_LLM(args)
    return model


def get_single_description(args, image_url, pipe):
    if args.Image_type == 'generated':
        image = Image.open(image_url)
    else:
        image = Image.open(os.path.join(args.data_path, args.test_set_images, image_url))
    env = Environment(loader=FileSystemLoader(args.prompt_path))
    template = env.get_template(args.MLLM_prompt)
    prompt = template.render()
    description = pipe(image, prompt=prompt, generate_kwargs={"max_new_tokens": 250})
    if isinstance(args.MLLM_prompt, str) and (
            'without_text' in args.MLLM_prompt or args.MLLM_prompt.strip() == 'description_generation.jinja'
    ):
        description = _sanitize_image_descriptor_output(description)
    return description


def get_combine_description(args, image_url, pipe):
    IN_descriptions = pd.read_csv(os.path.join(args.data_path,
                                               f'train/IN_LLAVA16_IN_description_generation_LLAVA16_description_PittAd.csv'))
    IN_description = IN_descriptions.loc[IN_descriptions['ID'] == image_url]['description'].values[0]
    UH_descriptions = pd.read_csv(os.path.join(args.data_path,
                                               f'train/IN_LLAVA16_UH_description_generation_llava16_description_PittAd.csv'))
    UH_description = UH_descriptions.loc[UH_descriptions['ID'] == image_url]['description'].values[0]
    v_descriptions = pd.read_csv(os.path.join(args.data_path,
                                              f'train/V_LLAVA16_v_description_generation_LLAVA16_description_PittAd.csv'))
    v_description = v_descriptions.loc[v_descriptions['ID'] == image_url]['description'].values[0]
    T_descriptions = pd.read_csv(os.path.join(args.data_path,
                                              f'train/T_LLAVA16_T_description_generation_LLAVA16_description_PittAd.csv'))
    T_description = T_descriptions.loc[T_descriptions['ID'] == image_url]['description'].values[0]
    data = {'IN': IN_description, 'UH': UH_description, 'v': v_description, 'T': T_description, 'token_length': None}
    env = Environment(loader=FileSystemLoader(args.prompt_path))
    template = env.get_template(args.MLLM_prompt)
    prompt = template.render(**data)
    description = pipe(prompt=prompt)
    return description


def get_descriptions(args, images):
    if args.task == 'whoops':
        images = [f'{i}.png' for i in range(500)]


    print(f'number of images in the set: {len(images)}')
    print('*' * 100)
    description_file = os.path.join(args.result_path,
                                    'results',
                                    args.project_name,
                                    f'{args.description_type}'
                                    f'_{args.MLLM}'
                                    f'_{"_".join(args.test_set_images.split("/")[-2:])}'
                                    f'_{args.AD_type}'
                                    f'_{args.MLLM_prompt.replace(".jinja", "")}.csv')
    if os.path.exists(description_file):
        print(f'{description_file} exists, reading the processed files')
        if args.resume:
            processed_images = set(pd.read_csv(description_file).ID.values)
            print(f'{len(processed_images)} images are processed and will be skipped')
        else:
            print(f'all {len(set(pd.read_csv(description_file).ID.values))} processed images will be overwritten')
            with open(description_file, 'w', newline='') as file:
                writer = csv.writer(file)
                # Write the header
                writer.writerow(['ID', 'description'])
            processed_images = set()
    else:
        print(f'{description_file} does not exist and is generated.')
        with open(description_file, 'w', newline='') as file:
            writer = csv.writer(file)
            # Write the header
            writer.writerow(['ID', 'description'])
        processed_images = set()
    pipe = get_model(args)
    print(f'image description generation started for {len(images) - len(processed_images)} images')
    for image_url in images:
        print(f'processing {image_url}')
        if image_url in processed_images:
            continue
        if args.Image_type != 'generated' and not os.path.exists(os.path.join(args.data_path, args.test_set_images, image_url)):
            continue
        processed_images.add(image_url)
        if args.description_type == 'combine':
            description = get_combine_description(args, image_url, pipe)
        else:
            description = get_single_description(args, image_url, pipe)
        print(f'description of image {image_url} is:\n {description}')
        print('-' * 80)
        image_ID = '/'.join(image_url.split('/')[-3:])
        pair = [image_ID, description]
        with open(description_file, 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(pair)

    return pd.read_csv(description_file)


def get_llm_generated_prompt(args, test_images):
    print(f'number of images in test set: {len(test_images)}')
    print('*' * 100)
    description_file = os.path.join(args.data_path,
                                    f'train/{args.llm_prompt.replace(".jinja", f"_{args.LLM}_FT{args.fine_tuned}")}_{args.AD_type}.csv')
    if os.path.exists(description_file):
        processed_images = set(pd.read_csv(description_file).ID.values)
        # return pd.read_csv(description_file)
    else:
        with open(description_file, 'w', newline='') as file:
            writer = csv.writer(file)
            # Write the header
            writer.writerow(['ID', 'description'])
        processed_images = set()
    prompt_generator = get_llm(args)
    for image_url in test_images:
        if image_url in processed_images:
            continue
        processed_images.add(image_url)
        description = prompt_generator.generate_prompt(args, image_url)
        print(f'output of image {image_url} is {description}')
        print('-' * 80)
        pair = [image_url, description]
        with open(description_file, 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(pair)

    return pd.read_csv(description_file)


def generate_description(args):
    # if args.AD_type == 'Sensation':
    #     test_images = []
    #     for sensation in SENSATIONS_PARENT_MAP:
    #         for i in range(10):
    #             test_images.append(f'{sensation}/{str(i)}.png')
    # else:
    if args.Image_type == 'generated':
        test_images = pd.read_csv(args.test_set_QA).generated_image_url.values
    else:
        test_images = get_test_data(args)
    if args.description_goal == 'image_descriptor':
        descriptions = get_descriptions(args, test_images)
    if args.description_goal == 'prompt_expansion':
        descriptions = get_llm_generated_prompt(args, test_images)

