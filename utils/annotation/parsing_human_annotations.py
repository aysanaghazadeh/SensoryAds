import pandas as pd
from utils.data.physical_sensations import SENSATION_HIERARCHY, SENSATIONS_PARENT_MAP
import json
# from configs.inference_config import get_args

def parse_sensation_annotations(annotation_file):
    sensation_annotation = pd.read_csv(annotation_file).values
    sensations_list = SENSATIONS_PARENT_MAP.keys()
    image_sensation_map = {}
    for row in sensation_annotation:
        image_url = row[0]
        print(f'Image: {image_url}')
        image_sensations_score = {}
        image_sensations_visual_elements = {}
        color_tone = set()
        scores = []
        for i in [21, 44, 67]:
            if str(row[i]) != 'nan':
                scores.append(row[i])
        if len(scores) > 0:
            max_score = max(scores)
        else:
            max_score = 0
        for sensation in sensations_list:
            image_sensations_score[sensation.lower()] = 0
            image_sensations_visual_elements[sensation.lower()] = set()

        if str(row[1]) == 'None':
            image_sensations_score['None'] = 5
            image_sensation_map[image_url] = {'sensation_scores': image_sensations_score,
                                              'visual_elements': list(image_sensations_visual_elements),
                                              'color_tone': list(color_tone),
                                              'image_sensations': ['no sensation']}
            continue
        most_evoked_sensation = []
        for i in range(1, 21):
            if str(row[i]) != 'nan':
                if int(row[21]) == max_score:
                    most_evoked_sensation.append(row[i].lower())
                row[i] = row[i].strip().lower()
                image_sensations_score[row[i]] = max(int(row[21]), image_sensations_score[row[i]])
                image_sensations_visual_elements[row[i]].add(row[22])
                color_tone.add(row[23])
        temp_sensation = []
        for i in range(24, 44):
            if str(row[i]) != 'nan':
                if int(row[21]) == max_score:
                    temp_sensation.append(row[i].lower())
                row[i] = row[i].strip().lower()
                image_sensations_score[row[i]] = max(int(row[44]), image_sensations_score[row[i]])
                image_sensations_visual_elements[row[i]].add(row[45])
                color_tone.add(row[46])
        if len(most_evoked_sensation) == 0:
            most_evoked_sensation = temp_sensation
        temp_sensation = []
        for i in range(47, 67):
            if str(row[i]) != 'nan':
                if int(row[21]) == max_score:
                    temp_sensation.append(row[i].lower())
                row[i] = row[i].strip().lower()
                image_sensations_score[row[i]] = max(int(row[67]), image_sensations_score[row[i]])
                image_sensations_visual_elements[row[i]].add(row[68])
                color_tone.add(row[69])
        if len(most_evoked_sensation) == 0:
            most_evoked_sensation = temp_sensation
        image_sensation_map[image_url] = {'sensation_scores': image_sensations_score,
                                          'visual_elements': list(image_sensations_visual_elements),
                                          'color_tone': list(color_tone),
                                          'image_sensations': most_evoked_sensation}
    json.dump(image_sensation_map, open(annotation_file.replace('.csv', '_parsed.json'), 'w'))

def get_SENSATIONS_PARENT_MAP(sensations, parent='root'):
    if sensations is None:
        return {}
    if isinstance(sensations, list):
        SENSATIONS_PARENT_MAP = {}
        for sense in sensations:
            SENSATIONS_PARENT_MAP[sense] = parent
        return SENSATIONS_PARENT_MAP
    if isinstance(sensations, dict):
        SENSATIONS_PARENT_MAP = {}
        for sense in sensations:
            SENSATIONS_PARENT_MAP[sense] = parent
            SENSATIONS_PARENT_MAP.update(get_SENSATIONS_PARENT_MAP(sensations[sense], sense))
        return SENSATIONS_PARENT_MAP

def get_human_human_annotations(human_annotations, sensations):
    human1_scores = {}
    human2_scores = {}
    for i in range(0, len(human_annotations), 2):
        human_annotations_1 = human_annotations[i]
        human_annotations_2 = human_annotations[i + 1]
        if human_annotations_1[0] != human_annotations_2[0]:
            print('not the same image')
        scores1, scores2 = {}, {}
        for sensation in sensations:
            scores1[sensation.lower()], scores2[sensation.lower()] = 0, 0
        for sensation_annotation in human_annotations_1[1:]:
            if str(sensation_annotation) == 'nan':
                continue
            if 'other' in sensation_annotation.lower():
                continue
            sensation_list, score = sensation_annotation.split('-')[:-3], sensation_annotation.split('-')[-3]
            for sensation in sensation_list:
                if sensation.lower() == 'motion and weight  sensation':
                    sensation = 'Motion and Weight'
                if sensation.lower() == 'atmospheric phenomena':
                    sensation = 'Atmospheric Phenomena Sound'
                if sensation.lower() == 'chemical and pungent':
                    sensation = 'chemical and pungent smell'
                if sensation.lower() == 'pungent':
                    sensation = 'pungent smell'
                if sensation.lower() == 'cooling minty tast':
                    sensation = 'Cooling Minty Taste'
                if sensation.lower() == 'sour':
                    sensation = 'Sour Taste'
                if sensation.lower() == 'texture sensation':
                    sensation = 'Texture'
                if sensation.lower() == 'splash':
                    sensation = 'Liquid Splash Sound'

                scores1[sensation.lower()] = max(scores1[sensation.lower()], int(score))
        for sensation_annotation in human_annotations_2[1:]:
            if str(sensation_annotation) == 'nan':
                continue
            if 'other' in sensation_annotation.lower():
                continue
            sensation_list, score = sensation_annotation.split('-')[:-3], sensation_annotation.split('-')[-3]
            for sensation in sensation_list:
                if sensation.lower() == 'motion and weight  sensation':
                    sensation = 'Motion and Weight'
                if sensation.lower() == 'atmospheric phenomena':
                    sensation = 'Atmospheric Phenomena Sound'
                if sensation.lower() == 'chemical and pungent':
                    sensation = 'chemical and pungent smell'
                if sensation.lower() == 'pungent':
                    sensation = 'pungent smell'
                if sensation.lower() == 'cooling minty tast':
                    sensation = 'Cooling Minty Taste'
                if sensation.lower() == 'sour':
                    sensation = 'Sour Taste'
                if sensation.lower() == 'texture sensation':
                    sensation = 'Texture'
                if sensation.lower() == 'splash':
                    sensation = 'Liquid Splash Sound'
                scores2[sensation.lower()] = max(scores2[sensation.lower()], int(score))
        human1_scores[human_annotations_1[0]] = scores1
        human2_scores[human_annotations_2[0]] = scores2
        json.dump(human1_scores, open(annotation_file.replace('.csv', 'human_1_parsed.json'), 'w'))
        json.dump(human2_scores, open(annotation_file.replace('.csv', 'human_2_parsed.json'), 'w'))
    return human1_scores, human2_scores


if __name__ == '__main__':

    # args = get_args()
    # sensation_parent_map = get_SENSATIONS_PARENT_MAP(SENSATION_HIERARCHY)
    # print(sensation_parent_map)
    # annotation_file = args.description_file
    annotation_file = '/Users/aysanaghazadeh/Downloads/human_human_data.csv'
    annotations = pd.read_csv(annotation_file).values
    # parse_sensation_annotations(annotation_file)
    human_score1, human_score2 = get_human_human_annotations(annotations, SENSATIONS_PARENT_MAP.keys())