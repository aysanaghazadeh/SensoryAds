import numpy as np
from utils.data.physical_sensations import SENSATIONS_PARENT_MAP

def compute_pearson_correlation(scores, human_annotations):
    correlation = np.corrcoef(scores, human_annotations)[0, 1]
    return correlation

def get_human_scores_per_image(human_annotations, image_url, sensation_list):
    human_scores = []
    for sensation in sensation_list:
        human_scores.append(human_annotations[image_url]['sensation_scores'][sensation])
    return human_scores

def get_scores_per_image(metric_scores, image_url, sensation_list):
    scores = []
    for sensation in sensation_list:
        score = metric_scores.loc[
            (metric_scores["ID"] == image_url) & (metric_scores["predicted_sensation"] == sensation),
            "average"
        ].iloc[0]
        scores.append(score)
    return scores

def get_human_score_agreement(metric_scores, human_annotations, args):
    return
