# from configs.inference_config import get_args
from utils.annotation.agreement import get_human_score_agreement
import json


if __name__ == '__main__':
    # args = get_args()
    # human_annotations = json.load(open(args.sensation_annotations))
    # metrics = json.load(open(args.description_file))
    human_annotations = json.load(open('/Users/aysanaghazadeh/Downloads/sensation_annotations_parsed.json'))
    metrics = json.load(open('/Users/aysanaghazadeh/experiments/SensoryAds/IN_InternVL_train_images_total_ALL_description_generation.json'))
    get_human_score_agreement(metrics, human_annotations)