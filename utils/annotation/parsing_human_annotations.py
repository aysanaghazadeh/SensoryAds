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
        for sensation in sensations_list:
            image_sensations_score[sensation.lower()] = 0
            image_sensations_visual_elements[sensation.lower()] = set()

        if str(row[1]) == 'None':
            image_sensations_score['None'] = 5
            continue

        for i in range(1, 21):
            if str(row[i]) != 'nan':
                row[i] = row[i].strip().lower()
                image_sensations_score[row[i]] = max(int(row[21]), image_sensations_score[row[i]])
                image_sensations_visual_elements[row[i]].add(row[22])
                color_tone.add(row[23])

        for i in range(24, 44):
            if str(row[i]) != 'nan':
                row[i] = row[i].strip().lower()
                image_sensations_score[row[i]] = max(int(row[44]), image_sensations_score[row[i]])
                image_sensations_visual_elements[row[i]].add(row[45])
                color_tone.add(row[46])
        for i in range(47, 67):
            if str(row[i]) != 'nan':
                row[i] = row[i].strip().lower()
                image_sensations_score[row[i]] = max(int(row[67]), image_sensations_score[row[i]])
                image_sensations_visual_elements[row[i]].add(row[68])
                color_tone.add(row[69])
        image_sensation_map[image_url] = {'sensation_scores': image_sensations_score,
                                          'visual_elements': list(image_sensations_visual_elements),
                                          'color_tone': list(color_tone)}
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
    # sensation_parent_map = get_SENSATIONS_PARENT_MAP(SENSATION_HIERARCHY)
    # print(sensation_parent_map)
    annotation_file = '/Users/aysanaghazadeh/Downloads/421Annotations.csv'
    parse_sensation_annotations(annotation_file)