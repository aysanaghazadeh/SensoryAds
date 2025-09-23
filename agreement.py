# from configs.inference_config import get_args
import numpy as np

from utils.annotation.agreement import get_human_score_agreement, get_kappa_agreement, get_krippendorff_agreement
import json


if __name__ == '__main__':
    # args = get_args()
    # human_annotations = json.load(open(args.sensation_annotations))
    # metrics = json.load(open(args.description_file))
    human_annotations = json.load(open('/Users/aysanaghazadeh/Downloads/sensation_annotations_parsed.json'))
    metrics = json.load(open('/Users/aysanaghazadeh/experiments/SensoryAds/IN_InternVL_train_images_total_ALL_description_generation_LLAMA3_instruct_finetunedTrue_12000.json'))
    # get_human_score_agreement(metrics, human_annotations)
    # get_kappa_agreement(metrics, human_annotations)
    # get_krippendorff_agreement(metrics, human_annotations)

    import numpy as np
    import math

    values = []
    for image_url in metrics:
        for sensation in metrics[image_url]:
            values.append(metrics[image_url][sensation][-1])
    #
    min_val, max_val = min(values), max(values)
    print('maximum', max(values))
    print('minimum', min(values))
    print('mean', sum(values) / len(values))
    print('median', np.median(values))
    print('stdev', np.std(values))
    # print((math.exp(metrics['2/39922.jpg']['Dryness'][-1]) - min_val) / (max_val - min_val))
    # print(1 / (1 + math.exp(-(metrics['2/39922.jpg']['Moisture and Dryness'][-1] - min_val) / (max_val - min_val))))
    print((metrics['2/39922.jpg']['Touch'][-1] - min_val) / (max_val - min_val))
    # data = metrics['2/39922.jpg']
    # print(data['Moisture and Dryness'])
    # sorted_dict = dict(sorted(data.items(), key=lambda item: item[1], reverse=True))
    # print(sorted_dict)