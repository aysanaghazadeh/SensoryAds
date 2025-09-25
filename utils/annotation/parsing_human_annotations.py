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



if __name__ == '__main__':

    # args = get_args()
    # sensation_parent_map = get_SENSATIONS_PARENT_MAP(SENSATION_HIERARCHY)
    # print(sensation_parent_map)
    # annotation_file = args.description_file
    annotation_file = '/Users/aysanaghazadeh/Downloads/gen_images_annotations.csv'
    parse_sensation_annotations(annotation_file)
