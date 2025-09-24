# from configs.inference_config import get_args
import numpy as np

from utils.annotation.agreement import get_human_score_agreement, get_kappa_agreement, get_krippendorff_agreement
import json


if __name__ == '__main__':
    # args = get_args()
    # human_annotations = json.load(open(args.sensation_annotations))
    # metrics = json.load(open(args.description_file))
    human_annotations = json.load(open('/Users/aysanaghazadeh/Downloads/sensation_annotations_parsed.json'))
    metrics = json.load(open('/Users/aysanaghazadeh/experiments/SensoryAds/IN_InternVL_train_images_total_ALL_description_generation_LLAMA3_instruct_finetunedTrue_21000.json'))
    # get_human_score_agreement(metrics, human_annotations)
    get_kappa_agreement(metrics, human_annotations)
    # get_krippendorff_agreement(metrics, human_annotations)
