# from configs.inference_config import get_args
import numpy as np

from utils.annotation.agreement import get_human_score_agreement, get_kappa_agreement, get_krippendorff_agreement
import json


if __name__ == '__main__':
    # args = get_args()
    # human_annotations = json.load(open(args.sensation_annotations))
    # metrics = json.load(open(args.description_file))

    human_annotations = json.load(open('/Users/aysanaghazadeh/Downloads/gen_images_annotations_parsed.json'))
    print('Evosense-LLAMA3-InternVL')
    metrics = json.load(open('/Users/aysanaghazadeh/experiments/SensoryAds/SensoryAds/Evosense_LLM_generated/gen_images_human_annotated_images.json'))
    get_human_score_agreement(metrics, human_annotations)
    get_kappa_agreement(metrics, human_annotations)
    # get_krippendorff_agreement(metrics, human_annotations)

    print('Evosense-QWenLM-InternVL')
    metrics = json.load(open(
        '/Users/aysanaghazadeh/experiments/SensoryAds/SensoryAds/Evosense_LLM_generated/gen_images_human_annotated_images_InternVL_QWenLM_isFineTunedTrue.json'))
    get_human_score_agreement(metrics, human_annotations)
    get_kappa_agreement(metrics, human_annotations)
    # get_krippendorff_agreement(metrics, human_annotations)

    print('Evosense-LLAMA3-QWenVL')
    metrics = json.load(open(
        '/Users/aysanaghazadeh/experiments/SensoryAds/SensoryAds/Evosense_LLM_generated/gen_images_human_annotated_images_QWenVL_LLAMA3_instruct_isFineTunedTrue.json'))
    get_human_score_agreement(metrics, human_annotations)
    get_kappa_agreement(metrics, human_annotations)
    # get_krippendorff_agreement(metrics, human_annotations)

    print('Evosense-QWenLM-QWenVL')
    metrics = json.load(open(
        '/Users/aysanaghazadeh/experiments/SensoryAds/SensoryAds/Evosense_LLM_generated/gen_images_human_annotated_images_QWenVL_QWenLM_isFineTunedTrue.json'))
    get_human_score_agreement(metrics, human_annotations)
    get_kappa_agreement(metrics, human_annotations)
    # get_krippendorff_agreement(metrics, human_annotations)

    print('Evosense-QWenLM-InternVL_Zeroshot')
    metrics = json.load(open(
        '/Users/aysanaghazadeh/experiments/SensoryAds/SensoryAds/Evosense_LLM_generated/gen_images_human_annotated_images_InternVL_QWenLM_isFineTunedFalse.json'))
    get_human_score_agreement(metrics, human_annotations)
    get_kappa_agreement(metrics, human_annotations)
    # get_krippendorff_agreement(metrics, human_annotations)

    print('Evosense-LLAMA3-InternVL_Zeroshot')
    metrics = json.load(open(
        '/Users/aysanaghazadeh/experiments/SensoryAds/SensoryAds/Evosense_LLM_generated/gen_images_human_annotated_images_InternVL_LLAMA3_instruct_isFineTunedFalse.json'))
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
