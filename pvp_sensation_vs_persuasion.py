"""Relate the highest EvoSense sensation score of PVP images to their persuasion score.

Run after scripts/evaluate_evosense_LLM_PVP.sh, which scores each generated description against every
sensation (evaluation.py --evaluation_type=Evosense_LLM) and writes a JSON keyed by image ID:
  {ID: {sensation: [total_logprob, last_token_logprob, normalized_average_logprob]}}
This takes each image's highest normalized score and correlates it with the PVP persuasion Score.

  python pvp_sensation_vs_persuasion.py --scores=../experiments/results/SensoryAds/Evosense_LLM/<file>.json
"""
import argparse
import re

import json
import pandas as pd
from scipy.stats import pearsonr, spearmanr

NORMALIZED_AVG_LOGPROB = 2  # index in the per-sensation list


def id_of(image_path):
    # same ID prepare_pvp_images.py wrote for description_generation (relative to image_files, last 3 components)
    image_path = re.sub(r'^(PVP/)?image_files/', '', image_path)
    return '/'.join(image_path.split('/')[-3:])


def load_pvp():
    from datasets import load_dataset
    # gated dataset: needs `huggingface-cli login` or HF_TOKEN
    ds = load_dataset('holi-lab/PVP')
    return pd.concat([ds[split].to_pandas() for split in ds], ignore_index=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--scores', required=True, help='JSON written by evaluation.py Evosense_LLM')
    parser.add_argument('--output', default=None, help='per-image CSV (default: next to --scores)')
    cli = parser.parse_args()

    scores = json.load(open(cli.scores))
    top = pd.DataFrame({
        'ID': list(scores),
        'top_sensation': [max(s, key=lambda k: s[k][NORMALIZED_AVG_LOGPROB]) for s in scores.values()],
        'sensation_score': [max(v[NORMALIZED_AVG_LOGPROB] for v in s.values()) for s in scores.values()],
    })

    pvp = load_pvp()
    pvp['ID'] = pvp.Image_Path.map(id_of)
    # one row per annotator rating; persuasion is averaged over annotators per image
    persuasion = pvp.groupby('ID').Score.mean().rename('persuasion_score').reset_index()
    df = top.merge(persuasion, on='ID')
    print(f'{len(df)} of {len(top)} scored images matched a PVP persuasion score')

    for name, corr in [('Pearson', pearsonr), ('Spearman', spearmanr)]:
        r, p = corr(df.sensation_score, df.persuasion_score)
        print(f'{name}: r={r:.4f}  p={p:.4g}')

    df['sensation_quartile'] = pd.qcut(df.sensation_score, 4, labels=['Q1 (lowest)', 'Q2', 'Q3', 'Q4 (highest)'])
    print('\nMean persuasion score by sensation-score quartile')
    print(df.groupby('sensation_quartile', observed=True).persuasion_score.agg(['count', 'mean']))
    print('\nMean persuasion score by top sensation')
    print(df.groupby('top_sensation').persuasion_score.agg(['count', 'mean']).sort_values('mean', ascending=False))

    df.to_csv(cli.output or cli.scores.replace('.json', '_vs_persuasion.csv'), index=False)
