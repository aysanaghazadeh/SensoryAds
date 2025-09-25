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
                        choices=['COM', 'PSA', 'ALL', 'Sensation'])
    parser.add_argument('--description_goal',
                        type=str,
                        default='prompt_expansion',
                        choices=['prompt_expansion', 'image_descriptor'])
    parser.add_argument('--description_type',
                        type=str,
                        default='IN',
                        help='Choose among IN, UH, combine')
    parser.add_argument('--model_type',
                        type=str,
                        choices=['LLM', 'MLLM'],
                        default='MLLM',
                        help='the model used for text generation tasks like evaluation, description generation, etc')
    parser.add_argument('--MLLM',
                        type=str,
                        default='InternVL')
    parser.add_argument('--MLLM_prompt',
                        type=str,)
    parser.add_argument('--LLM_prompt',
                         type=str,
                         help='LLM input prompt template file name.')
    parser.add_argument('--T2I_prompt',
                         type=str,
                         help='T2I input prompt template file name.')
    parser.add_argument('--with_sentiment',
                        type=bool,
                        default=False,
                        help='True if you want to include the ground truth sentiment in the prompt.')
    parser.add_argument('--with_topics',
                        type=bool,
                        default=False,
                        help='True if you want to include the ground truth topic in the prompt.')
    parser.add_argument('--with_audience',
                        type=bool,
                        default=False,
                        help='True if you want to include the detected audience by LLM in the prompt.')
    parser.add_argument('--with_physical_sensation',
                        type=bool,
                        default=True)
    parser.add_argument('--model_path',
                        type=str,
                        default='../models',
                        help='The path to trained models')
    parser.add_argument('--config_path',
                        type=str,
                        default=None,
                        help='The path to the config file if config_type is YAML')
    parser.add_argument('--result_path',
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
                        help='LLM chosen from: Mistral7B, phi, LLaMA3')
    parser.add_argument('--train',
                        type=bool,
                        default=False,
                        help='True if the LLM is being fine-tuned')
    parser.add_argument('--data_path',
                        type=str,
                        default='../Data/PittAd',
                        help='Path to the root of the data'
                        )
    parser.add_argument('--sensation_annotations',
                        type=str,
                        default='train/50_sensation_annotation_processed.json',
                        help='Path to the annotations relative to the data path.')
    parser.add_argument('--train_ratio',
                        type=float,
                        default=0.7,
                        help='the ratio of the train size to the dataset in train test split.'
                        )
    parser.add_argument('--test_size',
                        type=int,
                        default=1500,
                        help='number of example in the test-set, it must be smaller than the original test set')
    parser.add_argument('--train_set_QA',
                        type=str,
                        default='train/Action_Reason_statements.json',
                        help='If the model is fine-tuned, relative path to the train-set QA from root path')
    parser.add_argument('--train_set_images',
                        type=str,
                        default='train_images_total',
                        help='If the model is fine-tuned, relative path to the train-set Images from root path')
    parser.add_argument('--test_set_QA',
                        type=str,
                        default='train/QA_Combined_Action_Reason_train.json',
                        help='Relative path to the QA file for action-reasons from root path'
                        )
    parser.add_argument('--test_set_images',
                        type=str,
                        default='train_images_total',
                        help='Relative path to the original images for the test set from root')
    parser.add_argument('--test_set_sensation',
                        type=str,
                        default='train/sensation_annotation_parsed.json')
    parser.add_argument('--text_input_type',
                        type=str,
                        default='AR',
                        choices=['LLM', 'AR', 'original_description', 'Sensation'],
                        help='Type of the input text for T2I generation model. Choose from LLM_generated (Generating image with LLM), '
                             'AR (for action-reason),'
                             'original_description (for combine, VT, IN, and atypicality descriptions),'
                             'Sensation for generating sensory images regardless of action-reason statements.')
    parser.add_argument('--description_file',
                        type=str,
                        default=None,
                        help='Path to the description file for the T2I input.')
    parser.add_argument('--prompt_path',
                        type=str,
                        default='utils/prompt_engineering/prompts',
                        help='Path to the folder of prompts. Set the name of prompt files as: {text_input_type}.jinja')
    parser.add_argument('--fine_tuned',
                        type=bool,
                        default=False,
                        help='True if you want to use the fine-tuned model')
    parser.add_argument('--api_key',
                        type=str,
                        default=None,
                        help='api key for openai')
    parser.add_argument('--resume',
                        default=False,
                        type=bool,
                        help='True if continuing the process from saved file.')
    parser.add_argument('--evaluation_type',
                        default='sensation_extraction',
                        type=str,
                        choices=['VQA_score',
                                 'Image_Reward',
                                 'PickScore',
                                 'CLIPScore',
                                 'VQA_score_GT_Sensation',
                                 'Image_Reward_GT_Sensation',
                                 'PickScore_GT_Sensation',
                                 'CLIPScore_GT_Sensation',
                                 'Evosense_LLM',
                                 'Evosense_LLM_generated',
                                 'Evosense_MLLM',
                                 'Evosense_GT_Sensation',],
                        help='Choose the evaluation metric')
    parser.add_argument('--Image_type',
                        type=str,
                        default='real',
                        choices=['real', 'generated']
                        )
    parser.add_argument('--model_checkpoint',
                        type=str,
                        default='3000')
    parser.add_argument('--retrieval_type',
                        type=str,
                        default='multichoice',
                        choices=['multichoice', 'hierarchy', 'first_level'],
                        help='Retrieval type which can be each of the retrieval task types')
    parser.add_argument('--result_filename',
                        type=str)
    return parser.parse_args()


def get_args():
    args = parse_args()
    if args.config_type == 'YAML':
        args = set_conf(args.config_path)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args.device = device
    print("Arguments are:\n", args, '\n', '-'*40)
    return args




