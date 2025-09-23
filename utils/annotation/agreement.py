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
            if sensation1 == sensation2:
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
            if human_score_sensation1 == human_score_sensation2 or metric_score_sensation1 == metric_score_sensation2:
                continue
            if human_score_sensation1 > human_score_sensation2:
                human_preferences.append(0)
            else:
                human_preferences.append(1)
            if metric_score_sensation1 > metric_score_sensation2:
                metric_preferences.append(0)
            else:
                metric_preferences.append(1)
    return human_preferences, metric_preferences


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

def get_krippendorff_agreement(metric_scores, human_annotations):
    human_preferences = []
    metrics_preferences = []
    sensation_list = list(SENSATIONS_PARENT_MAP)
    count = 0
    for image_url in metric_scores:
        if image_url not in human_annotations or count > 100:
            continue
        count += 1
        if count < 47:
            continue
        human_preferences_per_image, metrics_preferences_per_image = get_preference_per_image(human_annotations, metric_scores, sensation_list, image_url)
        metrics_preferences += metrics_preferences_per_image
        human_preferences += human_preferences_per_image
        print(f'Krippendorff score for image {image_url} is', compute_krippendorff(metrics_preferences_per_image, human_preferences_per_image))
        print('-'*100)

    print(f'overall alpha agreement for {count} images is:', compute_krippendorff(metrics_preferences, human_preferences))


def get_kappa_agreement(metric_scores, human_annotations):
    human_preferences = []
    metrics_preferences = []
    sensation_list = list(SENSATIONS_PARENT_MAP)
    count = 0
    for image_url in metric_scores:
        if image_url not in human_annotations or count > 100:
            continue
        count += 1
        if count < 47:
            continue
        human_preferences_per_image, metrics_preferences_per_image = get_preference_per_image(human_annotations, metric_scores, sensation_list, image_url)
        metrics_preferences += metrics_preferences_per_image
        human_preferences += human_preferences_per_image
        print(f'Kappa score for image {image_url} is', compute_cohen_kappa(metrics_preferences_per_image, human_preferences_per_image))
        print('-'*100)

    print(f'overall kappa agreement for {count} images is:', compute_cohen_kappa(metrics_preferences, human_preferences))
