from os.path import exists

from Evaluation.sensation_alignment_metrics import *
from utils.data.physical_sensations import SENSATIONS_PARENT_MAP
import json
from PIL import Image
from configs.evaluation_config import get_args
import pandas as pd
import os

class SensationEvaluation:
    def __init__(self, args):
        self.args = args

        if self.args.evaluation_type == 'Evosense_LLM':
            from LLMs.LLM import LLM
            self.model = LLM(args)
        if self.args.evaluation_type == 'Evosense_MLLM':
            from MLLMs.MLLM import MLLM
            self.model = MLLM(args)
        if self.args.evaluation_type == 'VQA_score':
            import t2v_metrics
            self.model = t2v_metrics.VQAScore(model='clip-flant5-xxl')
        if self.args.evaluation_type == 'Image_Reward':
            import t2v_metrics
            self.model = t2v_metrics.ITMScore(model='image-reward-v1')
        if self.args.evaluation_type == 'PickScore':
            import t2v_metrics
            self.model = t2v_metrics.CLIPScore(model='pickscore-v1')
        if self.args.evaluation_type == 'CLIPScore':
            import t2v_metrics
            self.model = t2v_metrics.CLIPScore(model='openai:ViT-B-32')

    def evaluate_Evosense_LLM(self, args):
        descriptions = pd.read_csv(args.description_file)
        result_filename = args.description_file.replace('.csv', f'_{args.LLM}_finetuned{args.fine_tuned}{f"_{args.model_checkpoint}" if args.fine_tuned else ""}.json').split('/')[-1]
        directory_path = os.path.join(args.result_path, 'results', args.project_name, args.evaluation_type)
        os.makedirs(directory_path, exist_ok=True)
        result_file = os.path.join(directory_path, result_filename)
        if os.path.exists(result_file) and args.resume:
            scores = json.load(open(result_file))
        else:
            scores = {}
        for index, row in descriptions.iterrows():
            image_url = row.ID
            description = row.description
            scores[image_url] = {}
            for sensation in SENSATIONS_PARENT_MAP:
                if image_url in scores and sensation in scores[image_url]:
                    continue
                total_logprob,_, last_token_logprob, average_logprob = get_EvoSense_LLM(args, self.model, description, sensation)
                scores[image_url][sensation] = [total_logprob, last_token_logprob, average_logprob]
            print(image_url)
            print(json.dumps(scores[image_url], indent=4))
            json.dump(scores, open(result_file, 'w'))

    def evaluate_Evosense_MLLM(self, args):
        descriptions = pd.read_csv(args.description_file)
        result_filename = args.description_file.replace('.csv',
                                                        f'_{args.MLLM}_finetuned{args.fine_tuned}{f"_{args.model_checkpoint}" if args.fine_tuned else ""}.json').split(
            '/')[-1]
        directory_path = os.path.join(args.result_path, 'results', args.project_name, args.evaluation_type)
        os.makedirs(directory_path, exist_ok=True)
        result_file = os.path.join(directory_path, result_filename)
        scores = {}
        for row in descriptions.iterrows():
            print(row)
            image_url = row['ID']
            if args.Image_type == 'generated':
                image = Image.open(os.path.join(args.result_path, 'generated_images', args.project_name, image_url))
            else:
                image = Image.open(os.path.join(args.data_path, args.test_set_images, image_url))
            description = row['description']
            scores[image_url] = {}
            for sensation in SENSATIONS_PARENT_MAP:
                total_logprob,_, last_token_logprob, average_logprob = get_EvoSense_MLLM(args, self.model, image, sensation)
                scores[image_url][sensation] = [total_logprob, last_token_logprob, average_logprob]
            json.dump(scores, open(result_file, 'w'))

    def evaluate_T2V(self, args):
        descriptions = pd.read_csv(args.description_file)
        result_filename = args.description_file.replace('.csv', '.json').split('/')[-1]
        directory_path = os.path.join(args.result_path, 'results', args.project_name, args.evaluation_type)
        os.makedirs(directory_path, exist_ok=True)
        result_file = os.path.join(directory_path, result_filename)
        scores = {}
        if os.path.exists(result_file) and args.resume:
            scores = json.load(open(result_file))
            print(f'{result_file} already exists and {len(scores)} scores will be loaded.')
        for index, row in descriptions.iterrows():
            image_url = row['ID']
            if image_url in scores:
                print(f'{image_url} is already processed: {scores[image_url]}')
                continue
            if args.Image_type == 'generated':
                image = os.path.join(args.result_path, 'generated_images', args.project_name, image_url)
            else:
                image = os.path.join(args.data_path, args.test_set_images, image_url)
            scores[image_url] = {}
            for sensation in SENSATIONS_PARENT_MAP:
                score = get_T2V_score(args, self.model, image, sensation)
                scores[image_url][sensation] = score.item()
            json.dump(scores, open(result_file, 'w'))

    def evaluate(self, args):
        evaluation_name = 'evaluate_' + args.evaluation_type
        if evaluation_name in ['evaluate_VQA_score', 'evaluate_Image_Reward', 'evaluate_PickScore_score', 'evaluate_CLIPScore_score']:
            evaluation_name = 'evaluate_T2V'
        print(f'evaluation method: {evaluation_name}')
        evaluation_method = getattr(self, evaluation_name)
        evaluation_method(args)

if __name__ == '__main__':
    args = get_args()
    evaluation = SensationEvaluation(args)
    evaluation.evaluate(args)
