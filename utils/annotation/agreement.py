import functools
from scipy.stats import spearmanr
import numpy as np
import itertools
from utils.data.physical_sensations import SENSATIONS_PARENT_MAP, SENSATION_HIERARCHY
import krippendorff
from sklearn.metrics import cohen_kappa_score
import pandas as pd
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
            if human_score_sensation1 == human_score_sensation2:
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

ERRORs = {}
parent_error_count = {}
def get_human_human_preference_per_image(human1_annotations, human2_annotations, sensation_list, image_url):
    human2_preferences = []
    human1_preferences = []

    for sensation1 in sensation_list:
        for sensation2 in sensation_list:
            level_parents = ['root']
            if SENSATIONS_PARENT_MAP[sensation1] not in level_parents or SENSATIONS_PARENT_MAP[sensation2] not in level_parents:
                continue
            if sensation1.lower() == sensation2.lower():
                continue
            if 'sensation_scores' in human1_annotations[image_url]:
                human1_annotations[image_url] = human1_annotations[image_url]['sensation_scores']
            if 'sensation_scores' in human2_annotations[image_url]:
                human2_annotations[image_url] = human2_annotations[image_url]['sensation_scores']
            if (sensation1 == sensation2
                    or sensation1.lower() not in human1_annotations[image_url]
                    or sensation1.lower() not in human2_annotations[image_url]
                    or sensation2.lower() not in human2_annotations[image_url]
                    or sensation2.lower() not in human1_annotations[image_url]) :
                print(image_url, sensation1.lower(), sensation2.lower())
                print(human1_annotations[image_url])
                print('-' * 100)
                print(human2_annotations[image_url])
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
            if human1_preferences[-1] != human2_preferences[-1] and SENSATIONS_PARENT_MAP[sensation1] == SENSATIONS_PARENT_MAP[sensation2]:
                ERRORs[image_url] = [sensation1, sensation2, human1_preferences[-1], human2_preferences[-1]]
                # if SENSATIONS_PARENT_MAP[sensation1] not in parent_error_count:
                #     parent_error_count[SENSATIONS_PARENT_MAP[sensation1]] = 1
                # else:
                #     parent_error_count[SENSATIONS_PARENT_MAP[sensation1]] += 1
                if sensation1 not in parent_error_count:
                    parent_error_count[sensation1] = 1
                else:
                    parent_error_count[sensation1] += 1
                if sensation2 not in parent_error_count:
                    parent_error_count[sensation2] = 1
                else:
                    parent_error_count[sensation2] += 1
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

N_PERM = 10000
N_BOOT = 10000


def _kappa_from_confusion(conf):
    conf = np.asarray(conf, dtype=float)
    n = conf.sum((-1, -2))
    p_o = np.trace(conf, axis1=-2, axis2=-1) / n
    p_e = (conf.sum(-2) * conf.sum(-1)).sum(-1) / n ** 2
    with np.errstate(divide='ignore', invalid='ignore'):
        return (p_o - p_e) / (1 - p_e)


def _alpha_from_confusion(conf):
    # two coders, no missing values: alpha = 1 - (n - 1) * sum_{c != k} o_ck / sum_{c != k} n_c n_k
    conf = np.asarray(conf, dtype=float)
    coincidence = conf + np.swapaxes(conf, -1, -2)
    n_c = coincidence.sum(-2)
    n = n_c.sum(-1)
    expected = n ** 2 - (n_c ** 2).sum(-1)
    observed = coincidence.sum((-1, -2)) - np.trace(coincidence, axis1=-2, axis2=-1)
    with np.errstate(divide='ignore', invalid='ignore'):
        return 1 - (n - 1) * observed / expected


def _percentile_ci(boot_stats, ci):
    alpha = 100 - ci
    lower, upper = np.nanpercentile(boot_stats, [alpha / 2, 100 - alpha / 2])
    return lower, upper


def categorical_agreement_with_ci_and_p(stat_from_confusion, x, y, groups, n_boot=N_BOOT, n_perm=N_PERM, ci=95, random_state=0):
    """Agreement statistic from the x-vs-y confusion matrix, bootstrap CI over images (pairs from one image
    are dependent) and a two-sided permutation p-value for H0: no agreement between x and y."""
    rng = np.random.default_rng(random_state)
    x, y = np.asarray(x), np.asarray(y)
    labels, codes = np.unique(np.concatenate([x, y]), return_inverse=True)
    k = len(labels)
    cx, cy = codes[:len(x)], codes[len(x):]
    _, image_ids = np.unique(np.asarray(groups), return_inverse=True)
    n_images = image_ids.max() + 1
    per_image = np.bincount(image_ids * k * k + cx * k + cy, minlength=n_images * k * k).reshape(n_images, k, k)
    total = per_image.sum(0)
    observed = stat_from_confusion(total)

    weights = rng.multinomial(n_images, np.full(n_images, 1 / n_images), size=n_boot)
    boot_stats = stat_from_confusion(np.tensordot(weights, per_image, axes=(1, 0)))
    lower, upper = _percentile_ci(boot_stats, ci)

    # shuffling y gives a random table with the observed row/column totals: draw it row by row
    row_totals, col_totals = total.sum(1), total.sum(0)
    null_tables = np.empty((n_perm, k, k))
    for b in range(n_perm):
        remaining = col_totals.copy()
        for i in range(k - 1):
            null_tables[b, i] = rng.multivariate_hypergeometric(remaining, row_totals[i])
            remaining = remaining - null_tables[b, i].astype(remaining.dtype)
        null_tables[b, k - 1] = remaining
    null_stats = stat_from_confusion(null_tables)
    p_value = (np.sum(np.abs(null_stats) >= abs(observed)) + 1) / (n_perm + 1)
    return observed, (lower, upper), p_value


def pearson_agreement_with_ci_and_p(x, y, groups, n_boot=N_BOOT, n_perm=N_PERM, ci=95, random_state=0):
    rng = np.random.default_rng(random_state)
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    _, image_ids = np.unique(np.asarray(groups), return_inverse=True)
    n_images = image_ids.max() + 1
    sums = np.stack([np.bincount(image_ids, weights=w, minlength=n_images)
                     for w in (np.ones_like(x), x, y, x * x, y * y, x * y)], axis=1)

    def pearson_from_sums(s):
        n, sx, sy, sxx, syy, sxy = (s[..., i] for i in range(6))
        cov = sxy / n - sx * sy / n ** 2
        return cov / np.sqrt((sxx / n - (sx / n) ** 2) * (syy / n - (sy / n) ** 2))

    observed = pearson_from_sums(sums.sum(0))
    weights = rng.multinomial(n_images, np.full(n_images, 1 / n_images), size=n_boot)
    lower, upper = _percentile_ci(pearson_from_sums(weights @ sums), ci)

    zx, zy = (x - x.mean()) / x.std(), (y - y.mean()) / y.std()
    null_stats = np.array([np.mean(zx * rng.permutation(zy)) for _ in range(n_perm)])
    p_value = (np.sum(np.abs(null_stats) >= abs(observed)) + 1) / (n_perm + 1)
    return observed, (lower, upper), p_value


kappa_with_ci_and_p = functools.partial(categorical_agreement_with_ci_and_p, _kappa_from_confusion)
alpha_with_ci_and_p = functools.partial(categorical_agreement_with_ci_and_p, _alpha_from_confusion)


def report_agreement(name, count, agreement_fn, x, y, groups):
    observed, (lower, upper), p_value = agreement_fn(x, y, groups)
    floor = 1 / (N_PERM + 1)
    p_text = f'p<{floor:.1g}' if p_value <= floor else f'p={p_value:.2g}'
    print(f'overall {name} for {count} images is: {observed:.4f} (95% CI [{lower:.4f}, {upper:.4f}], {p_text})')
    return observed, (lower, upper), p_value


def get_human_score_agreement(metric_scores, human_annotations):
    human_scores_list = []
    metrics_score_list = []
    groups = []
    sensation_list = list(SENSATIONS_PARENT_MAP)
    count = 0
    for image_url in metric_scores:
        if image_url not in human_annotations:
            continue
        count += 1
        # if count < 40:
        #     continue
        human_scores_per_image = get_human_scores_per_image(human_annotations, image_url, sensation_list)
        metrics_scores_per_image = get_scores_per_image(metric_scores, image_url, sensation_list)

        human_scores_list += human_scores_per_image
        metrics_score_list += metrics_scores_per_image
        groups += [count] * len(human_scores_per_image)
    print(f'total number of image-sensation pair is {len(metrics_score_list)}')
    return report_agreement('correlation', count, pearson_agreement_with_ci_and_p,
                            metrics_score_list, human_scores_list, groups)


def get_human_human_score_agreement(human1_annotations, human2_annotations):
    human1_scores_list = []
    human2_scores_list = []

    sensation_list = list(SENSATIONS_PARENT_MAP)
    count = 0
    for image_url in human1_annotations:
        if image_url not in human2_annotations:
            continue
        count += 1
        human1_scores_per_image = get_human_scores_per_image(human1_annotations, image_url, sensation_list)
        human2_scores_per_image = get_scores_per_image(human2_annotations, image_url, sensation_list)

        human1_scores_list += human1_scores_per_image
        human2_scores_list += human2_scores_per_image


def get_krippendorff_agreement(metric_scores, human_annotations):
    human_preferences = []
    metrics_preferences = []
    groups = []
    sensation_list = list(SENSATIONS_PARENT_MAP)
    count = 0
    for image_url in metric_scores:
        if image_url not in human_annotations:
            continue
        count += 1
        human_preferences_per_image, metrics_preferences_per_image = get_preference_per_image(human_annotations, metric_scores, sensation_list, image_url)
        metrics_preferences += metrics_preferences_per_image
        human_preferences += human_preferences_per_image
        groups += [count] * len(human_preferences_per_image)

    print(len(metrics_preferences), len(human_preferences))
    return report_agreement('alpha agreement', count, alpha_with_ci_and_p,
                            metrics_preferences, human_preferences, groups)

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


def get_sensations(human_annotations, image_url):
    sensation_scores= human_annotations[image_url]['sensation_scores']
    sensations = [sensation for sensation in sensation_scores if sensation_scores[sensation] > 0]
    return sensations


def get_kappa_agreement(metric_scores, human_annotations):
    human_preferences = []
    metrics_preferences = []
    groups = []
    sensation_list = list(SENSATIONS_PARENT_MAP)
    count = 0
    # descriptions = pd.read_csv('/Users/aysanaghazadeh/experiments/results/SensoryAds/new_results/IN_InternVL_train_images_total_ALL_description_generation.csv')
    for image_url in metric_scores:
        if image_url not in human_annotations:
            continue
        # if count < 100:
        #     continue
        # description = descriptions.loc[descriptions['ID'] == image_url]['description'].values[0]
        sensations = get_sensations(human_annotations, image_url)
        sensations_splitted = []
        for sensation in sensations:
            sensations_splitted += sensation.split(' ')
        # if any(sensation.lower() in description.lower() for sensation in sensations):
        #     continue
        count += 1
        human_preferences_per_image, metrics_preferences_per_image = get_preference_per_image(human_annotations, metric_scores, sensation_list, image_url)
        metrics_preferences += metrics_preferences_per_image
        human_preferences += human_preferences_per_image
        groups += [count] * len(human_preferences_per_image)
    return report_agreement('kappa agreement', count, kappa_with_ci_and_p,
                            metrics_preferences, human_preferences, groups)

def get_human_human_kappa_agreement(human1_annotations, human2_annotations):
    human1_preferences = []
    human2_preferences = []
    sensation_list = list(SENSATIONS_PARENT_MAP)
    count = 0
    for image_url in human1_annotations:
        if image_url not in human2_annotations:
            print(image_url)
            continue
        count += 1

        human1_preferences_per_image, human2_preferences_per_image = get_human_human_preference_per_image(human1_annotations, human2_annotations, sensation_list, image_url)
        human1_preferences += human1_preferences_per_image
        human2_preferences += human2_preferences_per_image
    print(len(human1_preferences), len(human2_preferences))
    print(f'overall kappa agreement for {count} images is:', compute_cohen_kappa(human1_preferences, human2_preferences))
    print(ERRORs)
    print(parent_error_count)

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
        count += len(sensations_group)
        correct_count += image_correct_count
    print('accuracy of first chosen value is:', correct_count / count,
          'out of total images of ', len(human_annotations.keys() & metric_scores.keys()))