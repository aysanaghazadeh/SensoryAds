"""Relate the highest EvoSense sensation score of PVP images to their persuasion score.

Two stages, so the GPU part and the analysis can be run separately:
  score    - EvoSense-score the generated description of every PVP image against every sensation
             (descriptions come from inference.py description_generation, see --description_file;
             needs the same LLM args as evaluation.py, e.g. --LLM, --fine_tuned, --model_name).
  analyze  - take each image's highest sensation score and correlate it with Score.

  python pvp_sensation_vs_persuasion.py score   --description_file=<descriptions.csv> --pvp_scores=pvp_evosense.json --config_type=DEFAULT --LLM=LLAMA3_instruct ...
  python pvp_sensation_vs_persuasion.py analyze --pvp_scores=pvp_evosense.json
"""
import argparse
import json
import os
import sys

import pandas as pd

# Same normalization as SensationEvaluation.evaluate_Evosense_LLM (constants fit on the train images).
NORM_MIN, NORM_MAX = -38.970709800720215, -2.043711707713487


def id_of(image_path):
    return '/'.join(image_path.split('/')[-3:])


def load_pvp():
    from datasets import load_dataset
    # gated dataset: needs `huggingface-cli login` or HF_TOKEN
    ds = load_dataset('holi-lab/PVP')
    return pd.concat([ds[split].to_pandas() for split in ds], ignore_index=True)


def score(pvp, descriptions_path, out_path):
    from configs.evaluation_config import get_args
    from Evaluation.sensation_alignment_metrics import get_EvoSense_LLM
    from LLMs.LLM import LLM
    from utils.data.physical_sensations import SENSATIONS_PARENT_MAP

    args = get_args()
    model = LLM(args)
    scores = json.load(open(out_path)) if os.path.exists(out_path) else {}
    # description_generation stores ID as the last 3 components of the image path
    descriptions = pd.read_csv(descriptions_path).set_index('ID').description
    images = pvp.drop_duplicates('Image_Path')
    images = images[images.Image_Path.map(id_of).isin(descriptions.index)]
    print(f'{len(images)} PVP images have a generated description')
    for i, (_, row) in enumerate(images.iterrows()):
        per_sensation = scores.setdefault(row.Image_Path, {})
        description = descriptions[id_of(row.Image_Path)]
        for sensation in SENSATIONS_PARENT_MAP:
            if sensation in per_sensation:
                continue
            _, _, _, average_logprob = get_EvoSense_LLM(args, model, description, sensation)
            per_sensation[sensation] = (average_logprob - NORM_MIN) / (NORM_MAX - NORM_MIN)
        if i % 20 == 0:
            json.dump(scores, open(out_path, 'w'))
            print(f'{i}/{len(images)} images scored')
    json.dump(scores, open(out_path, 'w'))


def analyze(pvp, scores_path):
    from scipy.stats import pearsonr, spearmanr

    scores = json.load(open(scores_path))
    top = pd.DataFrame({
        'Image_Path': list(scores),
        'top_sensation': [max(s, key=s.get) for s in scores.values()],
        'sensation_score': [max(s.values()) for s in scores.values()],
    })
    # one row per annotator rating; persuasion is averaged over annotators per image
    persuasion = pvp.groupby('Image_Path').Score.mean().rename('persuasion_score').reset_index()
    df = top.merge(persuasion, on='Image_Path')
    print(f'{len(df)} images with both a sensation and a persuasion score')

    for name, corr in [('Pearson', pearsonr), ('Spearman', spearmanr)]:
        r, p = corr(df.sensation_score, df.persuasion_score)
        print(f'{name}: r={r:.4f}  p={p:.4g}')

    df['sensation_quartile'] = pd.qcut(df.sensation_score, 4, labels=['Q1 (lowest)', 'Q2', 'Q3', 'Q4 (highest)'])
    print('\nMean persuasion score by sensation-score quartile')
    print(df.groupby('sensation_quartile', observed=True).persuasion_score.agg(['count', 'mean']))
    print('\nMean persuasion score by top sensation')
    print(df.groupby('top_sensation').persuasion_score.agg(['count', 'mean']).sort_values('mean', ascending=False))

    df.to_csv(scores_path.replace('.json', '_vs_persuasion.csv'), index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['score', 'analyze'])
    parser.add_argument('--pvp_scores', default='pvp_evosense.json')
    parser.add_argument('--description_file', help='CSV (ID, description) written by description_generation')
    own, rest = parser.parse_known_args()
    sys.argv = [sys.argv[0]] + rest  # the remaining flags belong to evaluation_config.get_args

    pvp = load_pvp()
    if own.stage == 'score':
        score(pvp, own.description_file, own.pvp_scores)
    else:
        analyze(pvp, own.pvp_scores)
