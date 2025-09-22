import numpy as np
from utils.data.physical_sensations import SENSATIONS_PARENT_MAP

def compute_pearson_correlation(scores, human_annotations):
    correlation = np.corrcoef(scores, human_annotations)[0, 1]
    return correlation

def get_human_scores_per_image(human_annotations, image_url, sensation_list):
    human_scores = []
    for sensation in sensation_list:
        human_scores.append(human_annotations[image_url]['sensation_scores'][sensation.lower()])
    return human_scores

def get_scores_per_image(metric_scores, image_url, sensation_list):
    if isinstance(metric_scores, dict):
        scores = []
        for sensation in sensation_list:
            if sensation not in metric_scores[image_url]:
                score = 0.00001
                print(sensation)
            else:
                sensation_scores = metric_scores[image_url][sensation]
                if isinstance(sensation_scores, list):
                    score = sensation_scores[-1]
                else:
                    score = sensation_scores
            scores.append(score)
        return scores
    scores = []
    for sensation in sensation_list:
        score = metric_scores.loc[
            (metric_scores["ID"] == image_url) & (metric_scores["predicted_sensation"] == sensation),
            "average"
        ].iloc[0]
        scores.append(score)
    return scores

def get_human_score_agreement(metric_scores, human_annotations):
    human_scores_list = []
    metrics_score_list = []

    sensation_list = list(SENSATIONS_PARENT_MAP)
    count = 0
    for image_url in metric_scores:
        if image_url not in human_annotations or count > 100:
            continue
        count += 1
        if count < 47:
            continue
        human_scores_per_image = get_human_scores_per_image(human_annotations, image_url, sensation_list)
        metrics_scores_per_image = get_scores_per_image(metric_scores, image_url, sensation_list)
        print(f'agreement on image {image_url}', compute_pearson_correlation(metrics_scores_per_image, human_scores_per_image))
        human_scores_list += human_scores_per_image
        metrics_score_list += metrics_scores_per_image
        print('-'*100)
    print(f'overall agreement for {count} images is:', compute_pearson_correlation(metrics_score_list, human_scores_list))

