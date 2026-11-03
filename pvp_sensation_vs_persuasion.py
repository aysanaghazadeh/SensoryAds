"""Relate the highest EvoSense sensation score of PVP images to their persuasion score.

Run after scripts/evaluate_evosense_LLM_PVP.sh, which scores each generated description against every
sensation (evaluation.py --evaluation_type=Evosense_LLM) and writes a JSON keyed by image ID:
  {ID: {sensation: [total_logprob, last_token_logprob, normalized_average_logprob]}}
This takes each image's highest normalized score and correlates it with the PVP persuasion Score.

  python pvp_sensation_vs_persuasion.py --scores=../experiments/results/SensoryAds/Evosense_LLM/<file>.json

With --wandb_project it also reports to Weights & Biases: the correlations, a sensation-vs-persuasion scatter,
the quartile means, and a table of the sample images (with the image itself) so they can be browsed in the UI.

It also saves a contact sheet (<scores>_sample.png) of --sample images, evenly spaced from the lowest to the
highest sensation score, each captioned with its top sensation, sensation score and persuasion score.
"""
import argparse
import json
import os
import re
import textwrap

import pandas as pd
from PIL import Image, ImageDraw
from scipy.stats import pearsonr, spearmanr

NORMALIZED_AVG_LOGPROB = 2  # index in the per-sensation list


def id_of(image_path):
    # same ID prepare_pvp_images.py wrote for description_generation (relative to image_files, last 3 components)
    image_path = re.sub(r'^(PVP/)?image_files/', '', image_path)
    return '/'.join(image_path.split('/')[-3:])


def save_contact_sheet(df, image_root, out_path, cols=5, thumb=300, caption_h=70):
    """df rows are laid out in order, left to right, top to bottom (lowest sensation score first)."""
    rows = -(-len(df) // cols)
    sheet = Image.new('RGB', (cols * thumb, rows * (thumb + caption_h)), 'white')
    draw = ImageDraw.Draw(sheet)
    for i, row in enumerate(df.itertuples()):
        x, y = (i % cols) * thumb, (i // cols) * (thumb + caption_h)
        try:
            image = Image.open(os.path.join(image_root, row.ID)).convert('RGB')
            image.thumbnail((thumb, thumb))
            sheet.paste(image, (x + (thumb - image.width) // 2, y))
        except OSError as e:
            draw.text((x + 4, y + 4), f'cannot open image: {e}', fill='red')
        caption = (f'{row.top_sensation}\nsensation {row.sensation_score:.3f}  persuasion {row.persuasion_score:.2f}\n'
                   + textwrap.shorten(row.ID, 48, placeholder='...'))
        draw.text((x + 4, y + thumb + 4), caption, fill='black')
    sheet.save(out_path)


def load_pvp():
    from datasets import load_dataset
    # gated dataset: needs `huggingface-cli login` or HF_TOKEN
    ds = load_dataset('holi-lab/PVP')
    return pd.concat([ds[split].to_pandas() for split in ds], ignore_index=True)


def pick_sample(df, n):
    """n rows evenly spaced from the lowest to the highest sensation score."""
    ranked = df.sort_values('sensation_score').reset_index(drop=True)
    picks = [round(i * (len(ranked) - 1) / max(n - 1, 1)) for i in range(min(n, len(ranked)))]
    return ranked.iloc[picks]


def log_to_wandb(cli, df, quartiles, correlations, sample, sheet_path):
    import wandb  # pip install wandb; set WANDB_MODE=offline on nodes without internet, then `wandb sync`

    run = wandb.init(project=cli.wandb_project, name=cli.wandb_name, config={'scores_file': cli.scores})
    run.summary.update({'n_images': len(df), **correlations})

    per_image = df[['ID', 'top_sensation', 'sensation_score', 'persuasion_score']]
    scatter_table = wandb.Table(dataframe=per_image)
    quartile_table = wandb.Table(dataframe=quartiles.reset_index().astype({'sensation_quartile': str}))
    logged = {
        'sensation_vs_persuasion': wandb.plot.scatter(scatter_table, 'sensation_score', 'persuasion_score',
                                                      title='Highest sensation score vs persuasion score'),
        'persuasion_by_sensation_quartile': wandb.plot.bar(quartile_table, 'sensation_quartile', 'mean',
                                                           title='Mean persuasion by sensation-score quartile'),
    }
    if sample is not None:
        sample_table = wandb.Table(columns=['image', 'ID', 'top_sensation', 'sensation_score', 'persuasion_score'])
        for row in sample.itertuples():
            path = os.path.join(cli.image_root, row.ID)
            if os.path.exists(path):
                sample_table.add_data(wandb.Image(path), row.ID, row.top_sensation, row.sensation_score,
                                      row.persuasion_score)
        logged['sample_images'] = sample_table
        logged['sample_contact_sheet'] = wandb.Image(sheet_path)
    run.log(logged)
    run.finish()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--scores', required=True, help='JSON written by evaluation.py Evosense_LLM')
    parser.add_argument('--output', default=None, help='per-image CSV (default: next to --scores)')
    parser.add_argument('--sample', type=int, default=20, help='number of images in the sample (0 to skip)')
    parser.add_argument('--image_root', default='/ix/akovashka/aya34/projects/Data/PVP/image_files',
                        help='folder the image IDs are relative to')
    parser.add_argument('--wandb_project', default=None, help='report to this Weights & Biases project')
    parser.add_argument('--wandb_name', default=None, help='W&B run name')
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

    correlations = {}
    for name, corr in [('pearson', pearsonr), ('spearman', spearmanr)]:
        r, p = corr(df.sensation_score, df.persuasion_score)
        correlations.update({f'{name}_r': r, f'{name}_p': p})
        print(f'{name.capitalize()}: r={r:.4f}  p={p:.4g}')

    df['sensation_quartile'] = pd.qcut(df.sensation_score, 4, labels=['Q1 (lowest)', 'Q2', 'Q3', 'Q4 (highest)'])
    quartiles = df.groupby('sensation_quartile', observed=True).persuasion_score.agg(['count', 'mean'])
    print('\nMean persuasion score by sensation-score quartile')
    print(quartiles)
    print('\nMean persuasion score by top sensation')
    print(df.groupby('top_sensation').persuasion_score.agg(['count', 'mean']).sort_values('mean', ascending=False))

    df.to_csv(cli.output or cli.scores.replace('.json', '_vs_persuasion.csv'), index=False)

    sample = sheet_path = None
    if cli.sample and os.path.isdir(cli.image_root):
        sample = pick_sample(df, cli.sample)
        sheet_path = cli.scores.replace('.json', '_sample.png')
        save_contact_sheet(sample, cli.image_root, sheet_path)
        print(f'\nsaved contact sheet of {len(sample)} images to {sheet_path}')
    elif cli.sample:
        print(f'\n--image_root {cli.image_root} not found, skipping the image sample')

    if cli.wandb_project:
        log_to_wandb(cli, df, quartiles, correlations, sample, sheet_path)
