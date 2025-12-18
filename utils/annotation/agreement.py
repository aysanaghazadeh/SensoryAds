from scipy.stats import spearmanr
import numpy as np
from utils.data.physical_sensations import SENSATIONS_PARENT_MAP, SENSATION_HIERARCHY
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


def get_human_human_preference_per_image(human1_annotations, human2_annotations, sensation_list, image_url):
    human2_preferences = []
    human1_preferences = []

    for sensation1 in sensation_list:
        for sensation2 in sensation_list:
            if (sensation1 == sensation2
                    or sensation1.lower() not in human1_annotations[image_url]
                    or sensation1.lower() not in human2_annotations[image_url]
                    or sensation2.lower() not in human2_annotations[image_url]
                    or sensation2.lower() not in human1_annotations[image_url]) :
                continue
            human1_score_sensation1 = human1_annotations[image_url][sensation1.lower()]
            human1_score_sensation2 = human1_annotations[image_url][sensation2.lower()]

            human2_score_sensation1 = human2_annotations[image_url][sensation1.lower()]
            human2_score_sensation2 = human2_annotations[image_url][sensation2.lower()]
            # if human1_score_sensation1 == human1_score_sensation2 or human2_score_sensation1 == human2_score_sensation2:
            #     continue

            if human1_score_sensation1 > human1_score_sensation2:
                human1_preferences.append(0)
            elif human1_score_sensation1 < human1_score_sensation2:
                human1_preferences.append(1)
            else:
                human1_preferences.append(2)
            if human2_score_sensation1 > human2_score_sensation2:
                human2_preferences.append(0)
            elif human2_score_sensation1 < human2_score_sensation2:
                human2_preferences.append(1)
            else:
                human2_preferences.append(2)
    return human1_preferences, human2_preferences


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

def bootstrap_spearman(human_scores, metric_score, n_boot=10000, ci=95, random_state=None):
    rng = np.random.default_rng(random_state)
    n = len(human_scores)
    boot_stats = []
    human_scores = np.asarray(human_scores)
    metric_score = np.asarray(metric_score)
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)  # resample with replacement
        r, _ = spearmanr(human_scores[idx], metric_score[idx])
        boot_stats.append(r)

    lower = np.percentile(boot_stats, (100 - ci) / 2)
    upper = np.percentile(boot_stats, 100 - (100 - ci) / 2)

    r_obs, p_obs = spearmanr(human_scores, metric_score)
    return r_obs, (lower, upper), p_obs

def get_human_score_agreement(metric_scores, human_annotations):
    human_scores_list = []
    metrics_score_list = []
    sensation_list = list(SENSATIONS_PARENT_MAP)
    count = 0
    for image_url in metric_scores:
        if image_url not in human_annotations:
            continue
        count += 1
        # if count < 40:
        #     continue
        if count > 221:
            break
        human_scores_per_image = get_human_scores_per_image(human_annotations, image_url, sensation_list)
        metrics_scores_per_image = get_scores_per_image(metric_scores, image_url, sensation_list)
        
        human_scores_list += human_scores_per_image
        metrics_score_list += metrics_scores_per_image
    print(f'overall correlation for {count} images is:', compute_pearson_correlation(metrics_score_list, human_scores_list))
    print(f'total number of image-sensation pair is {len(metrics_score_list)}')


def get_human_human_score_agreement(human1_annotations, human2_annotations):
    human1_scores_list = []
    human2_scores_list = []

    sensation_list = list(SENSATIONS_PARENT_MAP)
    count = 0
    for image_url in human1_annotations:
        if image_url not in human2_annotations:
            continue
        count += 1
        # if count < 40:
        #     continue
        # if count > 140:
        #     break
        human1_scores_per_image = get_human_scores_per_image(human1_annotations, image_url, sensation_list)
        human2_scores_per_image = get_scores_per_image(human2_annotations, image_url, sensation_list)

        human1_scores_list += human1_scores_per_image
        human2_scores_list += human2_scores_per_image


def get_krippendorff_agreement(metric_scores, human_annotations):
    human_preferences = []
    metrics_preferences = []
    sensation_list = list(SENSATIONS_PARENT_MAP)
    count = 0
    for image_url in metric_scores:
        if image_url not in human_annotations:
            continue
        count += 1
        # if count < 40:
        #     continue
        # if count > 140:
        #     break
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


def bootstrap_kappa(rater1, rater2, n_boot=10000, ci=95, random_state=None, **kappa_kwargs):

    r1 = np.asarray(rater1)
    r2 = np.asarray(rater2)
    assert len(r1) == len(r2)

    rng = np.random.default_rng(random_state)
    n = len(r1)
    boot_stats = []

    for _ in range(n_boot):
        idx = rng.integers(0, n, n)  # resample rows with replacement
        k = cohen_kappa_score(r1[idx], r2[idx], **kappa_kwargs)
        boot_stats.append(k)

    k_obs = cohen_kappa_score(r1, r2, **kappa_kwargs)

    alpha = 100 - ci
    lower = np.percentile(boot_stats, alpha / 2)
    upper = np.percentile(boot_stats, 100 - alpha / 2)

    return k_obs, (lower, upper)


def get_kappa_agreement(metric_scores, human_annotations):
    human_preferences = []
    metrics_preferences = []
    sensation_list = list(SENSATIONS_PARENT_MAP)
    count = 0
    for image_url in metric_scores:
        if image_url not in human_annotations:
            continue
        count += 1
        # if count < 40:
        #     continue
        if count > 221:
            break

        human_preferences_per_image, metrics_preferences_per_image = get_preference_per_image(human_annotations, metric_scores, sensation_list, image_url)
        metrics_preferences += metrics_preferences_per_image
        human_preferences += human_preferences_per_image

    print(f'overall kappa agreement for {count} images is:', compute_cohen_kappa(metrics_preferences, human_preferences))


def get_human_human_kappa_agreement(human1_annotations, human2_annotations):
    human1_preferences = []
    human2_preferences = []
    sensation_list = list(SENSATIONS_PARENT_MAP)
    count = 0
    image_urls_redundunt = ['0/122910.jpg', '0/119030.jpg', '0/120020.jpg']
    for image_url in human1_annotations:
        if image_url not in human2_annotations or image_url in image_urls_redundunt:
            continue
        count += 1
        # if count < 40:
        #     continue
        # if count > 140:
        #     break

        human1_preferences_per_image, human2_preferences_per_image = get_human_human_preference_per_image(human1_annotations, human2_annotations, sensation_list, image_url)
        human1_preferences += human1_preferences_per_image
        human2_preferences += human2_preferences_per_image


    print(f'overall kappa agreement for {count} images is:', compute_cohen_kappa(human1_preferences, human2_preferences))


def get_first_sensation_accuracy(metric_scores, human_annotations):
    correct_count = 0
    count = 0
    for image_url in (metric_scores.keys() & human_annotations.keys()):
        image_metric_annotations = metric_scores[image_url]
        image_metric_annotations = dict(sorted(image_metric_annotations.items(), key=lambda x: x[1][-1], reverse=True))
        image_human_annotations = human_annotations[image_url]
        if sum(list(image_human_annotations['sensation_scores'].values())) == 0:
            human_sensation_chosen = ['none']
            continue
        else:
            human_sensation_chosen = [k for k, v in image_human_annotations['sensation_scores'].items() if v > 0]
        first_sensation = list(image_metric_annotations.keys())[0]
        second_sensation = list(image_metric_annotations.keys())[1]
        third_sensation = list(image_metric_annotations.keys())[2]
        if first_sensation.lower() in human_sensation_chosen:
            correct_count += 1
        # else:
        #     print(image_url)
        #     print(first_sensation, image_metric_annotations[first_sensation][-1])
        #     print(second_sensation, image_metric_annotations[second_sensation][-1])
        #     print(third_sensation, image_metric_annotations[third_sensation][-1])
        #     # if human_sensation_chosen == ['none']:
        #     print(human_sensation_chosen, image_metric_annotations['None'][-1])
        #
        #     print('-' * 30)
        count += 1
    print('accuracy of first chosen value is:', correct_count/count, 'out of total images of ', count)

def get_child_sensations(
        sensations
    ):
    if isinstance(sensations, list):
        return sensations
    elif isinstance(sensations, dict):
        return list(sensations.keys())
    else:
        return []

def get_image_metric_sensation_hierarchy(image_metric_scores, sensations, is_root=True, threshold=0.85):
    sensation_list = get_child_sensations(sensations)
    if len(sensation_list) == 0:
        return []
    current_level_sensation_scores = {key: image_metric_scores[key] for key in sensation_list if key in image_metric_scores}
    current_level_sensation_scores = dict(sorted(current_level_sensation_scores.items(), key=lambda x: x[1][-1], reverse=True))
    current_level_sensation_list = list(sorted(current_level_sensation_scores, key=lambda x: current_level_sensation_scores[x][-1], reverse=True))
    current_level_sensation = current_level_sensation_list[0]
    if isinstance(sensations, dict):
        image_metric_sensations_chosen = ([current_level_sensation] +
                                      get_image_metric_sensation_hierarchy(image_metric_scores, sensations[current_level_sensation]))
    else:
        image_metric_sensations_chosen = [current_level_sensation]
    if is_root:
        later_level_scores = [image_metric_scores[sensation][-1] for sensation in image_metric_sensations_chosen]
        later_level_scores = sum(later_level_scores) / len(later_level_scores)
    if is_root and later_level_scores < threshold:
        sensations_found = {current_level_sensation: [image_metric_sensations_chosen, image_metric_scores[image_metric_sensations_chosen[-1]][-1]]}
        sensation_idx = 1
        while sensation_idx < (len(current_level_sensation_scores) - 1) and later_level_scores < threshold:
            current_level_sensation = current_level_sensation_list[sensation_idx]
            if current_level_sensation == 'None':
                sensation_idx += 1
                continue
            if isinstance(sensations, dict):
                image_metric_sensations_chosen = ([current_level_sensation] +
                                                  get_image_metric_sensation_hierarchy(image_metric_scores,
                                                                                       sensations[current_level_sensation]))
            else:
                image_metric_sensations_chosen = [current_level_sensation]
            later_level_scores = [image_metric_scores[sensation][-1] for sensation in image_metric_sensations_chosen]
            later_level_scores = sum(later_level_scores) / len(later_level_scores)
            sensations_found[current_level_sensation] = [image_metric_sensations_chosen, later_level_scores]
            sensation_idx += 1
        if later_level_scores < threshold:
            max_score = 0
            for sensation_category in sensations_found:
                if sensations_found[sensation_category][1] >= max_score:
                    max_score = sensations_found[sensation_category][1]
                    image_metric_sensations_chosen = sensations_found[sensation_category][0]
    return image_metric_sensations_chosen


def get_hierarchy_first_sensation_agreement(metric_scores, human_annotations):
    correct_count = 0
    count = 0
    for image_url in metric_scores:
        if image_url not in human_annotations:
            continue
        image_metric_annotations = metric_scores[image_url]
        image_human_annotations = human_annotations[image_url]
        if sum(list(image_human_annotations['sensation_scores'].values())) == 0:
            human_sensation_chosen = ['none']
            continue
        else:
            human_sensation_chosen = [k for k, v in image_human_annotations['sensation_scores'].items() if v > 0]
        sensations_group = get_image_metric_sensation_hierarchy(image_metric_annotations, SENSATION_HIERARCHY, is_root=True)
        image_correct_count = 0
        for sensation in sensations_group:
            if sensation.lower() in human_sensation_chosen:
                image_correct_count += 1
        # if image_correct_count < len(sensations_group):
        #     print(image_url)
        #     print(sensations_group, [image_metric_annotations[sensation][-1] for sensation in sensations_group])
        #     print(human_sensation_chosen)
        #     print('-' * 30)
        count += len(sensations_group)
        correct_count += image_correct_count
    print('accuracy of first chosen value is:', correct_count / count,
          'out of total images of ', len(human_annotations.keys() & metric_scores.keys()))