import json

import numpy as np
from utils.data.physical_sensations import SENSATIONS_PARENT_MAP
import krippendorff
from sklearn.metrics import cohen_kappa_score

def compute_pearson_correlation(scores, human_annotations):
    correlation = np.corrcoef(scores, human_annotations)[0, 1]
    return correlation

def compute_krippendorff(scores, human_annotations):
    alpha = krippendorff.alpha(reliability_data=[scores, human_annotations], level_of_measurement='nominal')
    return alpha

def compute_cohen_kappa(scores, human_annotations):
    kappa = cohen_kappa_score(scores, human_annotations)
    return kappa

def get_human_scores_per_image(human_annotations, image_url, sensation_list):
    human_scores = []
    for sensation in sensation_list:
        human_scores.append(human_annotations[image_url]['sensation_scores'][sensation.lower()])
    return human_scores

def get_preference_per_image(human_annotations, metric_annotations, sensation_list, image_url):
    metric_preferences = []
    human_preferences = []
    for sensation1 in sensation_list:
        for sensation2 in sensation_list:
            if (sensation1 == sensation2
                    or sensation1.lower() not in human_annotations[image_url]['sensation_scores']
                    or sensation1 not in metric_annotations[image_url]
                    or sensation2 not in metric_annotations[image_url]
                    or sensation2.lower() not in human_annotations[image_url]['sensation_scores']) :
                continue
            human_score_sensation1 = human_annotations[image_url]['sensation_scores'][sensation1.lower()]
            human_score_sensation2 = human_annotations[image_url]['sensation_scores'][sensation2.lower()]

            sensation_scores1 = metric_annotations[image_url][sensation1]
            sensation_scores2 = metric_annotations[image_url][sensation2]
            if isinstance(sensation_scores1, list):
                metric_score_sensation1 = sensation_scores1[-1]
                metric_score_sensation2 = sensation_scores2[-1]
            else:
                metric_score_sensation1 = sensation_scores1
                metric_score_sensation2 = sensation_scores2
            if human_score_sensation1 == human_score_sensation2:# or metric_score_sensation1 == metric_score_sensation2:
                continue
            if human_score_sensation1 > human_score_sensation2:
                human_preferences.append(0)
            else:
                human_preferences.append(1)
            if metric_score_sensation1 > metric_score_sensation2:
                metric_preferences.append(0)
            elif metric_score_sensation1 < metric_score_sensation2:
                metric_preferences.append(1)
            else:
                metric_preferences.append(2)
    return human_preferences, metric_preferences


def get_scores_per_image(metric_scores, image_url, sensation_list):
    if isinstance(metric_scores, dict):
        scores = []
        for sensation in sensation_list:
            if sensation not in metric_scores[image_url]:
                score = 0.00001
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
        if image_url not in human_annotations:
            continue
        count += 1
        if count < 10:
            continue
        if count > 140:
            break
        human_scores_per_image = get_human_scores_per_image(human_annotations, image_url, sensation_list)
        metrics_scores_per_image = get_scores_per_image(metric_scores, image_url, sensation_list)
        # print(f'agreement on image {image_url}', compute_pearson_correlation(metrics_scores_per_image, human_scores_per_image))
        human_scores_list += human_scores_per_image
        metrics_score_list += metrics_scores_per_image
        if count == 12:
            print(image_url)
            print(compute_pearson_correlation(metrics_scores_per_image, human_scores_per_image))
    print(f'overall correlation for {count} images is:', compute_pearson_correlation(metrics_score_list, human_scores_list))

def get_krippendorff_agreement(metric_scores, human_annotations):
    human_preferences = []
    metrics_preferences = []
    sensation_list = list(SENSATIONS_PARENT_MAP)
    count = 0
    for image_url in metric_scores:
        if image_url not in human_annotations:
            continue
        count += 1
        if count < 40:
            continue
        if count > 140:
            break
        human_preferences_per_image, metrics_preferences_per_image = get_preference_per_image(human_annotations, metric_scores, sensation_list, image_url)
        metrics_preferences += metrics_preferences_per_image
        human_preferences += human_preferences_per_image

    print(f'overall alpha agreement for {count} images is:', compute_krippendorff(metrics_preferences, human_preferences))

def get_per_class_krippendorff_agreement(metric_scores, human_annotations):
    human_preferences = {
                             'touch': [],
                             'smell': [],
                             'sound': [],
                             'taste': [],
                             'sight': [],
                             'none': [],
                             'all': []
                        }
    metrics_preferences = {
                             'touch': [],
                             'smell': [],
                             'sound': [],
                             'taste': [],
                             'sight': [],
                             'none': [],
                             'all': []
                        }
    sensation_count = {
                         'touch': 0,
                         'smell': 0,
                         'sound': 0,
                         'taste': 0,
                         'sight': 0,
                         'none': 0,
                         'all': 0
                       }
    sensation_list = list(SENSATIONS_PARENT_MAP)
    count = -1
    for image_url in metric_scores:
        count += 1
        if count < 40:
            continue
        if count > 140:
            continue
        if image_url not in human_annotations:
            continue
        sensation_category = None
        maximum_score = max(human_annotations[image_url]['sensation_scores'].values())
        if maximum_score > 0:
            for sensation in human_preferences:
                if human_annotations[image_url]['sensation_scores'][sensation] == maximum_score:
                    sensation_category = sensation
                    sensation_count[sensation] += 1
                    break
        else:
            sensation_count['none'] += 1
            sensation_category = 'none'
            human_annotations[image_url]['sensation_scores']['none'] = 5

        human_preferences_per_image, metrics_preferences_per_image = get_preference_per_image(human_annotations, metric_scores, sensation_list, image_url)
        metrics_preferences[sensation_category] += metrics_preferences_per_image
        human_preferences[sensation_category] += human_preferences_per_image
        if sensation_category != 'none':
            metrics_preferences['all'] += metrics_preferences_per_image
            human_preferences['all'] += human_preferences_per_image
            sensation_count['all'] += 1

    for sensation in human_preferences:
        if sensation_count[sensation] > 0:
            print(f'overall alpha agreement for {sensation_count[sensation]} images evoking {sensation} sensation is:', compute_krippendorff(metrics_preferences[sensation], human_preferences[sensation]))


def get_kappa_agreement(metric_scores, human_annotations):
    human_preferences = []
    metrics_preferences = []
    sensation_list = list(SENSATIONS_PARENT_MAP)
    count = 0
    for image_url in metric_scores:
        if image_url not in human_annotations:
            continue
        count += 1
        if count < 10:
            continue
        if count > 140:
            break

        human_preferences_per_image, metrics_preferences_per_image = get_preference_per_image(human_annotations, metric_scores, sensation_list, image_url)
        metrics_preferences += metrics_preferences_per_image
        human_preferences += human_preferences_per_image
        # if compute_cohen_kappa(metrics_preferences_per_image, human_preferences_per_image) > 0.7:
        #     print(f'Kappa score for image {image_url} is', compute_cohen_kappa(metrics_preferences_per_image, human_preferences_per_image))
        # # print('-'*100)
        if count == 12:
            metric_human_scores = {}
            for sensation in metric_scores[image_url]:
                metric_human_scores[sensation] = [human_annotations[image_url]['sensation_scores'][sensation.lower()], metric_scores[image_url][sensation][-1]]
            # print(json.dumps(metric_human_scores, indent=4))
            print(compute_cohen_kappa(metrics_preferences_per_image, human_preferences_per_image))
    print(f'overall kappa agreement for {count} images is:', compute_cohen_kappa(metrics_preferences, human_preferences))
