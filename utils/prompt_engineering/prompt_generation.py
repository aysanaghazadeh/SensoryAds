import json
import os.path
from LLMs.LLM import LLM
import pandas as pd
from jinja2 import Environment, FileSystemLoader
from utils.data.mapping import SENTIMENT_MAP, TOPIC_MAP
from collections import Counter


class ImageGenerationPromptGenerator:
    def __init__(self, args):
        self.LLM_model = None
        self.descriptions = None
        self.sentiments = None
        self.topics = None
        self.audiences = None
        self.set_LLM(args)
        self.set_descriptions(args)
        self.set_sentiments(args)
        self.set_topics(args)
        self.set_audience(args)

    def set_LLM(self, args):
        if args.text_input_type == 'LLM':
            self.LLM_model = LLM(args)

    def set_descriptions(self, args):
        if args.text_input_type not in ['LLM', 'AR']:
            self.descriptions = self.get_all_descriptions(args)

    def set_sentiments(self, args):
        if args.with_sentiment:
            self.sentiments = self.get_all_sentiments(args)

    def set_topics(self, args):
        if args.with_sentiment:
            self.topics = self.get_all_topics(args)

    def set_audience(self, args):
        if args.with_audience:
            self.audiences = self.get_all_audience(args)
    
    

    @staticmethod
    def get_all_sentiments(args):
        if not args.with_sentiment:
            return None
        sentiment_file = os.path.join(args.data_path, 'train/Sentiments_train.json')
        sentiments = json.load(open(sentiment_file))
        return sentiments

    @staticmethod
    def get_all_topics(args):
        if not args.with_topics:
            return None
        topics_file = os.path.join(args.data_path, 'train/Topics_train.json')
        topics = json.load(open(topics_file))
        return topics

    @staticmethod
    def get_all_audience(args):
        if not args.with_audience:
            return None
        audience_file = os.path.join(args.data_path, 'train/get_audience.csv')
        audiences = pd.read_csv(audience_file)
        audiences = audiences.set_index('ID')['description'].to_dict()
        return audiences

    @staticmethod
    def get_all_descriptions(args):
        if args.text_input_type in ['AR', 'LLM']:
            return None
        descriptions = pd.read_csv(args.description_file)
        return descriptions

    @staticmethod
    def get_description(image_filename, descriptions):
        return descriptions.loc[descriptions['ID'] == image_filename]['description'].values[0]

    @staticmethod
    def get_LLM_input_prompt(args, action_reason, sentiment=None, topic=None, audience=None):
        data = {'action_reason': action_reason, 'sentiment': sentiment, 'topic': topic, 'audience': audience}
        env = Environment(loader=FileSystemLoader(args.prompt_path))
        template = env.get_template(args.llm_prompt)
        output = template.render(**data)
        return output

    @staticmethod
    def get_most_frequent(values):
        tuple_list = [tuple(inner_list) for inner_list in values]
        # Create a Counter object from the tuple list
        counter = Counter(tuple_list)
        # Get the most common tuple
        most_freq_tuple, _ = counter.most_common(1)[0]
        # Convert tuple back to list if necessary
        return list(most_freq_tuple)[0]

    def get_original_description_prompt(self, args, image_filename, sensation):
        QA_path = args.test_set_QA if not args.train else args.train_set_QA
        QA_path = os.path.join(args.data_path, QA_path)
        QA = json.load(open(QA_path))
        action_reason = QA[image_filename][0]
        sentiment = ''
        if args.with_sentiment:
            if image_filename in self.sentiments:
                sentiment_ids = self.sentiments[image_filename]
                sentiment_id = self.get_most_frequent(sentiment_ids)
                if sentiment_id in SENTIMENT_MAP:
                    sentiment = SENTIMENT_MAP[sentiment_id]
            else:
                print(f'there is no sentiment for image: {image_filename}')
        topic = ''
        if args.with_topics:
            if image_filename in self.topics:
                topic_ids = self.topics[image_filename]
                topic_id = self.get_most_frequent([topic_ids])
                if topic_id in TOPIC_MAP:
                    topic = TOPIC_MAP[topic_id]
            else:
                print(f'there is no topic for image: {image_filename}')
        audience = ''
        if args.with_audience:
            if image_filename in self.audiences:
                audience = self.audiences[image_filename]
                if len(audience.split(':')) > 1:
                    audience = audience.split(':')[-1].split('-')[-1]
                else:
                    audience = 'everyone'
            else:
                print(f'there is no audience for image: {image_filename}')
        
        if args.with_physical_sensation:
            physical_sensation = sensation if sensation else 'no sensation'
        
        data = {'action_reason': action_reason,
                'description': self.get_description(image_filename, self.descriptions).split('Description of the image:')[-1],
                'sentiment': sentiment,
                'topic': topic,
                'audience': audience,
                'sensation': physical_sensation}
        env = Environment(loader=FileSystemLoader(args.prompt_path))
        template = env.get_template(args.T2I_prompt)
        output = template.render(**data)
        return output

    def get_LLM_generated_prompt(self, args, image_filename, sensation):
        sentiment = ''
        if args.with_sentiment:
            if image_filename in self.sentiments:
                sentiment_ids = self.sentiments[image_filename]
                sentiment_id = self.get_most_frequent(sentiment_ids)
                if sentiment_id in SENTIMENT_MAP:
                    sentiment = SENTIMENT_MAP[sentiment_id]
            else:
                print(f'there is no sentiment for image: {image_filename}')
        topic = ''
        if args.with_topics:
            if image_filename in self.topics:
                topic_ids = self.topics[image_filename]
                topic_id = self.get_most_frequent([topic_ids])
                if topic_id in TOPIC_MAP:
                    topic = TOPIC_MAP[topic_id]
            else:
                print(f'there is no topic for image: {image_filename}')
        audience = ''
        if args.with_audience:
            if image_filename in self.audiences:
                audience = self.audiences[image_filename]
                if len(audience.split(':')) > 1:
                    audience = audience.split(':')[-1].split('-')[-1]
                else:
                    audience = 'everyone'
            else:
                print(f'there is no audience for image: {image_filename}')
        physical_sensation = ''
        if args.with_physical_sensation:
            physical_sensation = sensation if sensation else 'no sensation'
        
        QA_path = args.test_set_QA if not args.train else args.train_set_QA
        QA_path = os.path.join(args.data_path, QA_path)
        QA = json.load(open(QA_path))
        action_reason = QA[image_filename][0]
        
        LLM_input_prompt = self.get_LLM_input_prompt(args, action_reason, sentiment, topic, audience)
        description = self.LLM_model(LLM_input_prompt)
        # description = f'{description}'
        if 'objects:' in description:
            objects = description.split('objects:')[1]
            description = description.split('objects:')[0]
        else:
            objects = None
        if 'Adjective:' in description:
            adjective = description.split('Adjective:')[1]
            description = description.split('Adjective:')[0]
        else:
            adjective = None
        data = {'description': description,
                'action_reason': action_reason,
                'objects': objects,
                'adjective': adjective,
                'sentiment': sentiment,
                'topic': topic,
                'audience': audience,
                'sensation': physical_sensation}

        print('data:', data)
        env = Environment(loader=FileSystemLoader(args.prompt_path))
        template = env.get_template(args.T2I_prompt)
        output = template.render(**data)
        print('LLM generated prompt:', output)
        return output

    def get_AR_prompt(self, args, image_filename, sensation):
        sentiment = ''
        if args.with_sentiment:
            if image_filename in self.sentiments:
                sentiment_ids = self.sentiments[image_filename]
                sentiment_id = self.get_most_frequent(sentiment_ids)
                if sentiment_id in SENTIMENT_MAP:
                    sentiment = SENTIMENT_MAP[sentiment_id]
            else:
                print(f'there is no sentiment for image: {image_filename}')
        topic = ''
        if args.with_topics:
            if image_filename in self.topics:
                topic_ids = self.topics[image_filename]
                topic_id = self.get_most_frequent([topic_ids])
                if topic_id in TOPIC_MAP:
                    topic = TOPIC_MAP[topic_id]
            else:
                print(f'there is no topic for image: {image_filename}')
        audience = ''
        if args.with_audience:
            if image_filename in self.audiences:
                audience = self.audiences[image_filename]
                if len(audience.split(':')) > 1:
                    audience = audience.split(':')[-1].split('-')[-1]
                else:
                    audience = 'everyone'
            else:
                print(f'there is no audience for image: {image_filename}')
        physical_sensation = sensation if sensation else 'no sensation'
        QA_path = args.test_set_QA if not args.train else args.train_set_QA
        QA_path = os.path.join(args.data_path, QA_path)
        QA = json.load(open(QA_path))
        action_reason = QA[image_filename][0]
        
        data = {'action_reason': action_reason, 'sentiment': sentiment, 'topic': topic, 'audience': audience, 'sensation': physical_sensation}
        env = Environment(loader=FileSystemLoader(args.prompt_path))
        template = env.get_template(args.T2I_prompt)
        output = template.render(**data)
        print('AR prompt:', output)
        return output

    def get_sensation_prompt(self, args, image_filename, sensation):
        data = {'sensation': sensation}
        env = Environment(loader=FileSystemLoader(args.prompt_path))
        template = env.get_template(args.T2I_prompt)
        output = template.render(**data)
        print('sensation prompt:', output)
        return output

    def generate_prompt(self, args, image_filename, sensation=None):
        prompt_generator_name = f'get_{args.text_input_type}_prompt'
        print('method: ', prompt_generator_name)
        if prompt_generator_name == 'get_LLM_prompt':
            prompt_generator_name = 'get_LLM_generated_prompt'
        prompt_generation_method = getattr(self, prompt_generator_name)
        prompt = prompt_generation_method(args, image_filename, sensation)
        return prompt

def generate_text_generation_prompt(args, data):
    env = Environment(loader=FileSystemLoader(args.prompt_path))
    prompt_file = args.MLLM_prompt if args.model_type == 'MLLM' else args.LLM_prompt
    template = env.get_template(prompt_file)
    output = template.render(**data)
    return output