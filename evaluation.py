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

    def evaluate_Evosense_LLM(self, args):
        descriptions = pd.read_csv(args.description_file)
        result_filename = args.description_file.replace('.csv', '.json')
        result_file = os.path.join(args.results_path, args.project_name, args.evaluation_type, result_filename)
        scores = {}
        for row in descriptions.iterrows():
            image_url = row['ID']
            description = row['description']
            scores[image_url] = {}
            for sensation in SENSATIONS_PARENT_MAP:
                total_logprob,_, last_token_logprob, average_logprob = get_EvoSense_LLM(args, self.model, description, sensation)
                scores[image_url][sensation] = [total_logprob, last_token_logprob, average_logprob]
            json.dump(scores, open(result_file, 'w'))

    def evaluate_Evosense_MLLM(self, args):
        descriptions = pd.read_csv(args.description_file)
        result_filename = args.description_file.replace('.csv', '.json')
        result_file = os.path.join(args.results_path, args.project_name, args.evaluation_type, result_filename)
        scores = {}
        for row in descriptions.iterrows():
            image_url = row['ID']
            if args.Image_type == 'generated':
                image = Image.open(os.path.join(args.results_path, 'generated_images', args.project_name, image_url))
            else:
                image = Image.open(os.path.join(args.data_path, args.test_set_images, image_url))
            description = row['description']
            scores[image_url] = {}
            for sensation in SENSATIONS_PARENT_MAP:
                total_logprob,_, last_token_logprob, average_logprob = get_EvoSense_MLLM(args, self.model, image, sensation)
                scores[image_url][sensation] = [total_logprob, last_token_logprob, average_logprob]
            json.dump(scores, open(result_file, 'w'))


    def evaluate(self, args):
        print(args.evaluation_type)
        evaluation_name = 'evaluate_' + args.evaluation_type
        print(f'evaluation method: {evaluation_name}')
        evaluation_method = getattr(self, evaluation_name)
        evaluation_method(args)

if __name__ == '__main__':
    args = get_args()
    evaluation = SensationEvaluation(args)
    evaluation.evaluate(args)
