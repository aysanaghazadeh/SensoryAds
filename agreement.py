# from configs.inference_config import get_args
import numpy as np

from utils.annotation.agreement import get_human_score_agreement, get_kappa_agreement, get_krippendorff_agreement
import json


if __name__ == '__main__':
    # args = get_args()
    # human_annotations = json.load(open(args.sensation_annotations))
    # metrics = json.load(open(args.description_file))
    print('Evosense-LLAMA3-InternVL')
    human_annotations = json.load(open('/Users/aysanaghazadeh/Downloads/gen_images_annotations_parsed.json'))
    metrics = json.load(open('/Users/aysanaghazadeh/experiments/SensoryAds/SensoryAds/Evosense_LLM_generated/gen_images_human_annotated_images.json'))
    get_human_score_agreement(metrics, human_annotations)
    get_kappa_agreement(metrics, human_annotations)
    # get_krippendorff_agreement(metrics, human_annotations)
    print('VQAScore')
    metrics = json.load(open('/Users/aysanaghazadeh/experiments/SensoryAds/SensoryAds/VQA_score_generated/gen_images_human_annotated_images.json'))
    get_human_score_agreement(metrics, human_annotations)
    get_kappa_agreement(metrics, human_annotations)
    # get_krippendorff_agreement(metrics, human_annotations)

    print('PickScore')
    metrics = json.load(open('/Users/aysanaghazadeh/experiments/SensoryAds/SensoryAds/PickScore_generated/gen_images_human_annotated_images.json'))
    get_human_score_agreement(metrics, human_annotations)
    get_kappa_agreement(metrics, human_annotations)
    # get_krippendorff_agreement(metrics, human_annotations)

    print('Image-Reward')
    metrics = json.load(open('/Users/aysanaghazadeh/experiments/SensoryAds/SensoryAds/Image_Reward_generated/gen_images_human_annotated_images.json'))
    get_human_score_agreement(metrics, human_annotations)
    get_kappa_agreement(metrics, human_annotations)
    # get_krippendorff_agreement(metrics, human_annotations)

    print('CLIP-score')
    metrics = json.load(open('/Users/aysanaghazadeh/experiments/SensoryAds/SensoryAds/CLIPScore_generated/gen_images_human_annotated_images.json'))
    get_human_score_agreement(metrics, human_annotations)
    get_kappa_agreement(metrics, human_annotations)
    # get_krippendorff_agreement(metrics, human_annotations)
