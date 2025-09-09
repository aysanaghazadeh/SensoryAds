import torch
import yaml
import argparse


def read_yaml_config(file_path):
    """Reads a YAML configuration file and returns a dictionary of the settings."""
    with open(file_path, 'r') as file:
        return yaml.safe_load(file)


def convert_to_args(config):
    """Converts a nested dictionary to a list of command-line arguments."""

    args = {}
    for section, settings in config.items():
        for key, value in settings.items():
            args[key] = value
    return args


def set_conf(config_file):
    yaml_file_path = config_file
    config = read_yaml_config(yaml_file_path)
    args = convert_to_args(config)
    return args


def parse_args():
    """ Parsing the Arguments for the Advertisement Generation Project"""
    parser = argparse.ArgumentParser(description="Configuration arguments for advertisement generation:")
    parser.add_argument('--config_type',
                        type=str,
                        required=True,
                        help='Choose among ARGS for commandline arguments, DEFAULT for default values, or YAML for '
                             'config file')
    parser.add_argument('--config_path',
                        type=str,
                        default=None,
                        help='The path to the config file if config_type is YAML')
    parser.add_argument('--AD_type',
                        type=str,
                        default='COM',
                        choices=['COM', 'PSA', 'ALL'])
    parser.add_argument('--description_goal',
                        type=str,
                        default='prompt_expansion',
                        choices=['prompt_expansion', 'image_descriptor'])
    parser.add_argument('--model_path',
                        type=str,
                        default='../models',
                        help='The path to trained models')
    parser.add_argument('--results',
                        type=str,
                        default='../experiments',
                        help='The path to the folder for saving the results')
    parser.add_argument('--T2I_model',
                        type=str,
                        default='PixArt',
                        help='T2I generation model chosen from: PixArt, Expressive, ECLIPSE, Translate')
    parser.add_argument('--LLM',
                        type=str,
                        default='LLAMA3_instruct',
                        help='LLM chosen from: Mixtral7B, Mistral7B, Vicuna, LLaMA2')
    parser.add_argument('--train',
                        type=bool,
                        default=True,
                        help='True if the LLM is being fine-tuned')
    parser.add_argument('--data_path',
                        type=str,
                        default='../Data/PittAd',
                        help='Path to the root of the data'
                        )
    parser.add_argument('--train_ratio',
                        type=float,
                        default=0.4,
                        help='the ratio of the train size to the dataset in train test split.'
                        )
    parser.add_argument('--train_set_QA',
                        type=str,
                        default=None,
                        help='If the model is fine-tuned, relative path to the train-set QA from root path')
    parser.add_argument('--train_set_images',
                        type=str,
                        default=None,
                        help='If the model is fine-tuned, relative path to the train-set Images from root path')
    parser.add_argument('--test_set_QA',
                        type=str,
                        default='train/QA_Combined_Action_Reason_train.json',
                        help='Relative path to the QA file for action-reasons from root path'
                        )
    parser.add_argument('--test_set_images',
                        type=str,
                        default='train_images',
                        help='Relative path to the original images for the test set from root')
    parser.add_argument('--text_input_type',
                        type=str,
                        default='AR',
                        help='Type of the input text for T2I generation model. Choose from LLM_generated, '
                             'AR (for action-reason),'
                             'original_description (for combine, VT, IN, and atypicality descriptions)')
    parser.add_argument('--description_file',
                        type=str,
                        default=None,
                        help='Path to the description that includes only product name.')
    parser.add_argument('--product_file',
                        type=str,
                        default=None,
                        help='Path to the negative adjective for the action reason statements.')
    parser.add_argument('--negative_file',
                        type=str,
                        default=None,
                        help='Path to the description file for the T2I input.')
    parser.add_argument('--prompt_path',
                        type=str,
                        default='utils/prompt_engineering/prompts',
                        help='Path to the folder of prompts. Set the name of prompt files as: {text_input_type}.jinja')
    parser.add_argument('--llm_prompt',
                         type=str,
                         default='LLM_input.jinja',
                         help='LLM input prompt template file name.')
    parser.add_argument('--T2I_prompt',
                         type=str,
                         default='LLM.jinja',
                         help='T2I input prompt template file name.')
    parser.add_argument('--epochs',
                        type=int,
                        default=100,
                        help='Defines the number of epochs if train is True')
    parser.add_argument('--lr',
                        type=int,
                        default=5e-5,
                        help='learning rate for training the model')
    parser.add_argument('--batch_size',
                        type=int,
                        default=2,
                        help='batch size in the training process')
    parser.add_argument('--weight_decay',
                        type=int,
                        default=0.01)
    parser.add_argument('--VLM',
                        type=str)
    parser.add_argument('--evaluation_type',
                        type=str,
                        default='image_reward')
    parser.add_argument('--sensation_annotations',
                        type=str,
                        default='train/50_sensation_annotation_processed.json',
                        help='Path to the annotations relative to the data path.')
    parser.add_argument('--project_name',
                        type=str,
                        default='SensoryAds',
                        help='Your project name, it will be used to save the results')
    parser.add_argument('--task',
                        type=str,
                        default='PittAd',
                        help='Choose between PittAd, whoops')
    parser.add_argument('--AD_type',
                        type=str,
                        default='ALL',
                        choices=['COM', 'PSA', 'ALL'])
    parser.add_argument('--model_type',
                        type=str,
                        choices=['LLM', 'MLLM'],
                        default='MLLM',
                        help='the model used for text generation tasks like evaluation, description generation, etc')
    parser.add_argument('--MLLM',
                        type=str,
                        default='InternVL')
    parser.add_argument('--MLLM_prompt',
                        type=str, )
    parser.add_argument('--LLM_prompt',
                        type=str,
                        help='LLM input prompt template file name.')
    parser.add_argument('--T2I_prompt',
                        type=str,
                        help='T2I input prompt template file name.')
    parser.add_argument('--sensation_annotations',
                        type=str,
                        default='train/50_sensation_annotation_processed.json',
                        help='Path to the annotations relative to the data path.')
    parser.add_argument('--api_key',
                        type=str,
                        default=None,
                        help='api key for openai')
    parser.add_argument('--resume',
                        default=False,
                        type=bool,
                        help='True if continuing the process from saved file.')
    parser.add_argument('--inference_type',
                        default='sensation_extraction',
                        type=str,
                        choices=['image_generation', 'sensation_extraction', 'description_generation'],
                        help='Inference type which can be image generation, sensation_extraction, etc.')
    parser.add_argument('--result_filename',
                        type=str)
    parser.add_argument('--training_type',
                        type=str,
                        default='CPO_train',
                        choices=['CPO_train_LLM', 'DPO_train_MLLM'])
    return parser.parse_args()


def get_args():
    args = parse_args()
    if args.config_type == 'YAML':
        args = set_conf(args.config_path)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if torch.cuda.is_available():
        print(f"Number of GPUs available: {torch.cuda.device_count()}")
        print(f"devices: {[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]}")
    args.device = device
    print("Arguments are:\n", args, '\n', '-'*40)
    return args



