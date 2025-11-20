import random
from accelerate import PartialState
from utils.data.trian_test_split import get_train_data
from transformers import AutoTokenizer
from datasets import Dataset, Features, Sequence, Value, Array3D
import torchvision.transforms as transforms
from PIL import Image
import pandas as pd
import os
import json
import torch
from torch.utils.data import DataLoader
from torchvision.transforms.functional import crop
import numpy as np
import cv2
import mediapipe as mp
from PIL import Image, ImageDraw
from pathlib import Path

from PIL import Image
from PIL.ImageOps import exif_transpose
import itertools

from utils.data.mapping import *

class DreamBoothDataset(Dataset):
    """
    A dataset to prepare the instance and class images with the prompts for fine-tuning the model.
    It pre-processes the images.
    """

    def __init__(
            self,
            args,
            size=1024,
            repeats=1,
            center_crop=False,
    ):
        def load_train_data(args):
            train_set_images = get_train_data(args)
            print(f"Total training images available: {len(train_set_images)}")
            QAs = json.load(open(os.path.join(args.data_path, args.test_set_QA)))
            sensations = json.load(open(os.path.join(args.data_path, args.sensation_annotations)))
            dataset = {
                'image': [],
                'positive_text': [],
                'negative_text': []
            }
            negative_QAs = {}
            negative_QAs['reason'] = json.load(
                open(os.path.join(args.data_path, 'train/reason_hard_QA_Combined_Action_Reason_train.json')))

            for image_url in train_set_images:
                if image_url in sensations:
                    AR = '\n-'.join(QAs[image_url][0])
                    for sensation in sensations[image_url]['image_sensations']:
                        dataset['image'].append(
                            Image.open(os.path.join(args.data_path, args.train_set_images, image_url)))
                        prompt = f"""Generate an advertisement image that evokes {sensation} sensation and coveys the following messages:"""
                        positive = f'{prompt}\n{AR}'
                        negative = f'Not persuasive'
                        dataset['positive_text'].append(positive)
                        dataset['negative_text'].append(negative)

            print(f"Final dataset size: {len(dataset['image'])} samples")
            return dataset

        self.size = size
        self.center_crop = center_crop

        self.args = args
        self.custom_instance_prompts = None

        # if --dataset_name is provided or a metadata jsonl file is provided in the local --instance_data directory,
        # we load the training data using load_dataset

        self.instance_data_root = Path(args.data_path)
        if not self.instance_data_root.exists():
            raise ValueError("Instance images root doesn't exists.")
        dataset = load_train_data(args)
        self.custom_instance_prompts = dataset['positive_text']
        self.instance_positive_text = dataset['positive_text']
        self.instance_negative_text = dataset['negative_text']
        self.instance_images = []
        for img in dataset['image']:
            self.instance_images.extend(itertools.repeat(img, repeats))

        self.pixel_values = []
        train_resize = transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR)
        train_crop = transforms.CenterCrop(size) if center_crop else transforms.RandomCrop(size)
        train_flip = transforms.RandomHorizontalFlip(p=1.0)
        train_transforms = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )
        for image in self.instance_images:
            image = exif_transpose(image)
            if not image.mode == "RGB":
                image = image.convert("RGB")
            image = train_resize(image)
            if args.random_flip and random.random() < 0.5:
                # flip
                image = train_flip(image)
            if args.center_crop:
                y1 = max(0, int(round((image.height - args.resolution) / 2.0)))
                x1 = max(0, int(round((image.width - args.resolution) / 2.0)))
                image = train_crop(image)
            else:
                y1, x1, h, w = train_crop.get_params(image, (args.resolution, args.resolution))
                image = crop(image, y1, x1, h, w)
            image = train_transforms(image)
            self.pixel_values.append(image)

        self.num_instance_images = len(self.instance_images)
        self._length = self.num_instance_images

        self.image_transforms = transforms.Compose(
            [
                transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.CenterCrop(size) if center_crop else transforms.RandomCrop(size),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

    def __len__(self):
        return self._length

    def __getitem__(self, index):
        example = {}
        # Handle both single index and list of indices
        if isinstance(index, (list, tuple)):
            instance_images = [self.pixel_values[i] for i in index]
            instance_prompts = [self.instance_positive_text[i] for i in index]
            negative_prompts = [self.instance_negative_text[i] for i in index]
            example["instance_images"] = torch.stack(instance_images)
            example["instance_prompt"] = instance_prompts
            example["negative_prompt"] = negative_prompts
        else:
            instance_image = self.pixel_values[index]
            example["instance_images"] = instance_image
            example["instance_prompt"] = self.instance_positive_text[index]
            example["negative_prompt"] = self.instance_negative_text[index]
        return example


def collate_fn(examples, with_prior_preservation=False):
    pixel_values = [example["instance_images"] for example in examples]
    prompts = [example["instance_prompt"] for example in examples]
    negative_prompts = [example["negative_prompt"] for example in examples]
    # Concat class and instance examples for prior preservation.
    # We do this to avoid doing two forward passes.
    if with_prior_preservation:
        pixel_values += [example["class_images"] for example in examples]
        prompts += [example["class_prompt"] for example in examples]
        negative_prompts += [example["negative_prompt"] for example in examples]
    pixel_values = torch.stack(pixel_values)
    pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()

    batch = {"pixel_values": pixel_values, "prompts": prompts, "negative_prompts": negative_prompts}
    return batch


class PromptDataset(Dataset):
    "A simple dataset to prepare the prompts to generate class images on multiple GPUs."

    def __init__(self, prompt, num_samples):
        self.prompt = prompt
        self.num_samples = num_samples

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        example = {}
        example["prompt"] = self.prompt
        example["index"] = index
        return example
