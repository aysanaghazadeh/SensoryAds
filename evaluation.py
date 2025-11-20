from os.path import exists
from utils.data.mapping import image_list as human_annotated_gen_images
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
        if self.args.evaluation_type == 'Evosense_GT_Sensation':
            from LLMs.LLM import LLM
            self.model = LLM(args)
        if self.args.evaluation_type == 'Evosense_LLM_generated':
            from LLMs.LLM import LLM
            self.model = LLM(args)
        if self.args.evaluation_type == 'Evosense_MLLM':
            from MLLMs.MLLM import MLLM
            self.model = MLLM(args)
        if 'VQA_score' in self.args.evaluation_type:
            import t2v_metrics
            self.model = t2v_metrics.VQAScore(model='clip-flant5-xxl', cache_dir=os.getenv('HF_HOME'))
        if 'Image_Reward' in self.args.evaluation_type:
            import t2v_metrics
            self.model = t2v_metrics.ITMScore(model='image-reward-v1', cache_dir=os.getenv('HF_HOME'))
        if 'PickScore' in self.args.evaluation_type:
            import t2v_metrics
            self.model = t2v_metrics.CLIPScore(model='pickscore-v1', cache_dir=os.getenv('HF_HOME'))
        if 'CLIPScore' in self.args.evaluation_type:
            import t2v_metrics
            self.model = t2v_metrics.CLIPScore(model='openai:ViT-B-32', cache_dir=os.getenv('HF_HOME'))
        if 'MLLM' in self.args.evaluation_type:
            from MLLMs.MLLM import MLLM
            self.model = MLLM(args)

    def evaluate_Evosense_LLM(self, args):
        descriptions = pd.read_csv(args.description_file)
        result_filename = args.description_file.replace('.csv', f'_{args.LLM}_finetuned{args.fine_tuned}{f"_{args.model_checkpoint}" if args.fine_tuned else ""}.json').split('/')[-1]
        directory_path = os.path.join(args.result_path, 'results', args.project_name, args.evaluation_type)
        os.makedirs(directory_path, exist_ok=True)
        result_file = os.path.join(directory_path, result_filename)
        if os.path.exists(result_file) and args.resume:
            scores = json.load(open(result_file))
            print(f'{result_file} already exists and {len(scores)} images are processed and will be skipped.')
        else:
            if os.path.exists(result_file):
                scores = json.load(open(result_file))
                print(f'{result_file} already exists, and {len(scores)} images will be overwritten.')
            else:
                print(f'{result_file} does not exist and will be created.')
            scores = {}
        for index, row in descriptions.iterrows():
            image_url = row.ID
            description = row.description
            scores[image_url] = {}
            for sensation in SENSATIONS_PARENT_MAP:
                if image_url in scores and sensation in scores[image_url]:
                    continue
                total_logprob,_, last_token_logprob, average_logprob = get_EvoSense_LLM(args, self.model, description, sensation)
                average_logprob = (average_logprob - (-38.970709800720215)) / (-2.043711707713487 - (-38.970709800720215)) #the values are to normalize the log probabilities based on the train images.
                scores[image_url][sensation] = [total_logprob, last_token_logprob, average_logprob]
            print(image_url)
            print(json.dumps(scores[image_url], indent=4))
            json.dump(scores, open(result_file, 'w'))

    def evaluate_Evosense_LLM_generated(self, args):
        descriptions = pd.read_csv(args.description_file)
        result_filename = f'gen_images_human_annotated_images_{args.MLLM}_{args.LLM}_isFineTuned{args.fine_tuned}.json'
        directory_path = os.path.join(args.result_path, 'results', args.project_name, args.evaluation_type)
        os.makedirs(directory_path, exist_ok=True)
        result_file = os.path.join(directory_path, result_filename)
        if os.path.exists(result_file) and args.resume:
            scores = json.load(open(result_file))
            print(f'{result_file} already exists and {len(scores)} images are processed and will be skipped.')
        else:
            if os.path.exists(result_file):
                scores = json.load(open(result_file))
                print(f'{result_file} already exists, and {len(scores)} images will be overwritten.')
            else:
                print(f'{result_file} does not exist and will be created.')
            scores = {}
        for index, row in descriptions.iterrows():
            image_url = row.ID
            if image_url not in human_annotated_gen_images:
                continue
            description = row.description
            model_name = args.T2I_model
            image_url = '_'.join([model_name, image_url.split('/')[-1]])
            if image_url not in scores:
                scores[image_url] = {}
            for sensation in SENSATIONS_PARENT_MAP:
                if image_url in scores and sensation in scores[image_url]:
                    continue
                total_logprob,_, last_token_logprob, average_logprob = get_EvoSense_LLM(args, self.model, description, sensation)
                average_logprob = (average_logprob - (-38.970709800720215)) / (-2.043711707713487 - (-38.970709800720215)) #the values are to normalize the log probabilities based on the train images.
                scores[image_url][sensation] = [total_logprob, last_token_logprob, average_logprob]
            print(image_url)
            print(json.dumps(scores[image_url], indent=4))
            json.dump(scores, open(result_file, 'w'))

    def evaluate_Evosense_GT_Sensation(self, args):
        descriptions = pd.read_csv(args.description_file)
        result_filename = args.description_file.replace('.csv',
                                                        f'_{args.LLM}_finetuned{args.fine_tuned}{f"_{args.model_checkpoint}" if args.fine_tuned else ""}.json').split(
            '/')[-1]
        directory_path = os.path.join(args.result_path, 'results', args.project_name, args.evaluation_type)
        os.makedirs(directory_path, exist_ok=True)
        result_file = os.path.join(directory_path, result_filename)
        if os.path.exists(result_file) and args.resume:
            scores = json.load(open(result_file))
            print(f'{result_file} already exists and {len(scores)} images are processed and will be skipped.')
        else:
            if os.path.exists(result_file):
                scores = json.load(open(result_file))
                print(f'{result_file} already exists, and {len(scores)} images will be overwritten.')
            else:
                print(f'{result_file} does not exist and will be created.')
            scores = {}
        for index, row in descriptions.iterrows():
            image_url = '/'.join(row.ID.split('/')[-2:])
            sensation = row.ID.split('/')[0]
            if args.AD_type == 'Sensation':
                sensation = row.ID.split('/')[1]
            description = row.description.split('Q2:')[-1]
            if image_url not in scores:
                scores[image_url] = {}
            if image_url in scores and sensation in scores[image_url]:
                continue
            total_logprob, _, last_token_logprob, average_logprob = get_EvoSense_LLM(args, self.model, description,
                                                                                     sensation)
            average_logprob = (average_logprob - (-38.970709800720215)) / (-2.043711707713487 - (
                -38.970709800720215))  # the values are to normalize the log probabilities based on the train images.
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
        if os.path.exists(result_file) and args.resume:
            scores = json.load(open(result_file))
            print(f'{result_file} already exists and {len(scores)} images are processed and will be skipped.')
        else:
            if os.path.exists(result_file):
                scores = json.load(open(result_file))
                print(f'{result_file} already exists, and {len(scores)} images will be overwritten.')
            else:
                print(f'{result_file} does not exist and will be created.')
            scores = {}
        for row in descriptions.iterrows():
            image_url = row['ID']
            if args.Image_type == 'generated':
                image = Image.open(os.path.join(args.result_path, 'generated_images', args.project_name, image_url))
            else:
                image = Image.open(os.path.join(args.data_path, args.test_set_images, image_url))
            description = row['description']
            if image_url not in scores:
                scores[image_url] = {}
            for sensation in SENSATIONS_PARENT_MAP:
                if image_url in scores and sensation in scores[image_url]:
                    continue
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
            print(f'{image_url} \n {json.dumps(scores[image_url], indent=4)}')
            json.dump(scores, open(result_file, 'w'))

    def evaluate_T2V_GT_Sensation(self, args):
        descriptions = pd.read_csv(args.description_file)
        result_filename = args.description_file.replace('.csv', '.json').split('/')[-1]
        directory_path = os.path.join(args.result_path, 'results', args.project_name, args.evaluation_type)
        os.makedirs(directory_path, exist_ok=True)
        result_file = os.path.join(directory_path, result_filename)
        if os.path.exists(result_file) and args.resume:
            scores = json.load(open(result_file))
            print(f'{result_file} already exists and {len(scores)} images are processed and will be skipped.')
        else:
            if os.path.exists(result_file):
                scores = json.load(open(result_file))
                print(f'{result_file} already exists, and {len(scores)} images will be overwritten.')
            else:
                print(f'{result_file} does not exist and will be created.')
            scores = {}
        for index, row in descriptions.iterrows():
            image_url = '/'.join(row.ID.split('/')[-2:])
            sensation = row.ID.split('/')[0]
            if args.AD_type == 'Sensation':
                sensation = row.ID.split('/')[1]
            if image_url in scores and sensation in scores[image_url]:
                continue
            if args.Image_type == 'generated':
                image = os.path.join(args.result_path, 'generated_images', args.project_name, args.test_set_images, row.ID)
            else:
                image = os.path.join(args.data_path, args.test_set_images, row.ID)
            if image_url not in scores:
                scores[image_url] = {}
            score = get_T2V_score(args, self.model, image, sensation)
            scores[image_url][sensation] = score.item()
            print(f'{image_url} \n {json.dumps(scores[image_url], indent=4)}')
            json.dump(scores, open(result_file, 'w'))

    def evaluate_T2V_generated(self, args):
        descriptions = pd.read_csv(args.description_file)
        result_filename = 'gen_images_human_annotated_images.json'
        directory_path = os.path.join(args.result_path, 'results', args.project_name, args.evaluation_type)
        os.makedirs(directory_path, exist_ok=True)
        result_file = os.path.join(directory_path, result_filename)
        if os.path.exists(result_file) and args.resume:
            scores = json.load(open(result_file))
            print(f'{result_file} already exists and {len(scores)} images are processed and will be skipped.')
        else:
            if os.path.exists(result_file):
                scores = json.load(open(result_file))
                print(f'{result_file} already exists, and {len(scores)} images will be overwritten.')
            else:
                print(f'{result_file} does not exist and will be created.')
            scores = {}
        for index, row in descriptions.iterrows():
            image_url = row.ID
            if image_url not in human_annotated_gen_images:
                continue
            model_name = args.T2I_model
            image_url = '_'.join([model_name, image_url.split('/')[-1]])
            for sensation in SENSATIONS_PARENT_MAP:
                if image_url in scores and sensation in scores[image_url]:
                    print(image_url)
                    continue
                image = os.path.join(args.result_path, 'generated_images', args.project_name, args.test_set_images, row.ID)
                if image_url not in scores:
                    scores[image_url] = {}
                score = get_T2V_score(args, self.model, image, sensation)
                scores[image_url][sensation] = score.item()
                print(f'{image_url} \n {json.dumps(scores[image_url], indent=4)}')
                json.dump(scores, open(result_file, 'w'))

    def evaluate_MLLM_GT_Sensation(self, args):
        descriptions = pd.read_csv(args.description_file)
        result_filename = args.description_file.replace('.csv', '.json').split('/')[-1]
        directory_path = os.path.join(args.result_path, 'results', args.project_name, args.evaluation_type)
        os.makedirs(directory_path, exist_ok=True)
        result_file = os.path.join(directory_path, result_filename)
        if os.path.exists(result_file) and args.resume:
            scores = json.load(open(result_file))
            print(f'{result_file} already exists and {len(scores)} images are processed and will be skipped.')
        else:
            if os.path.exists(result_file):
                scores = json.load(open(result_file))
                print(f'{result_file} already exists, and {len(scores)} images will be overwritten.')
            else:
                print(f'{result_file} does not exist and will be created.')
            scores = {}
        for index, row in descriptions.iterrows():
            image_url = '/'.join(row.ID.split('/')[-2:])
            sensation = row.ID.split('/')[0]
            if args.AD_type == 'Sensation':
                sensation = row.ID.split('/')[1]
            if image_url in scores and sensation in scores[image_url]:
                continue
            if args.Image_type == 'generated':
                image = os.path.join(args.result_path, 'generated_images', args.project_name, args.test_set_images,
                                     row.ID)
            else:
                image = os.path.join(args.data_path, args.test_set_images, row.ID)
            if image_url not in scores:
                scores[image_url] = {}
            score = get_MMLM_Judge_Score(args, self.model, image, sensation)
            scores[image_url][sensation] = score
            print(f'{image_url} \n {json.dumps(scores[image_url], indent=4)}')
            json.dump(scores, open(result_file, 'w'))

    def evaluate_MLLM(self, args):
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
                score = get_MMLM_Judge_Score(args, self.model, image, sensation)
                scores[image_url][sensation] = score
            # print(f'{image_url} \n {json.dumps(scores[image_url], indent=4)}')
            json.dump(scores, open(result_file, 'w'))

    def evaluate(self, args):
        evaluation_name = 'evaluate_' + args.evaluation_type
        if evaluation_name in ['evaluate_VQA_score', 'evaluate_Image_Reward', 'evaluate_PickScore', 'evaluate_CLIPScore']:
            evaluation_name = 'evaluate_T2V'
        elif evaluation_name in ['evaluate_VQA_score_GT_Sensation', 'evaluate_Image_Reward_GT_Sensation', 'evaluate_PickScore_GT_Sensation', 'evaluate_CLIPScore_GT_Sensation']:
            evaluation_name = 'evaluate_T2V_GT_Sensation'
        elif evaluation_name.split('_generated')[0] in ['evaluate_VQA_score', 'evaluate_Image_Reward', 'evaluate_PickScore',
                               'evaluate_CLIPScore']:
            evaluation_name = 'evaluate_T2V_generated'
        print(f'evaluation method: {evaluation_name}')
        evaluation_method = getattr(self, evaluation_name)
        evaluation_method(args)

if __name__ == '__main__':
    args = get_args()
    evaluation = SensationEvaluation(args)
    evaluation.evaluate(args)
