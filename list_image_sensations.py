"""List every scored image with its top sensations, from the JSON written by evaluation.py Evosense_LLM.

  python list_image_sensations.py --scores=<Evosense_LLM json> [--top_k 3] [--output images_sensations.csv]

Needs only the standard library. Columns: ID, sensation (highest), score, then the top_k sensations with scores.
"""
import argparse
import csv
import json

NORMALIZED_AVG_LOGPROB = 2  # index in each sensation's [total_logprob, last_token_logprob, normalized_average_logprob]

parser = argparse.ArgumentParser()
parser.add_argument('--scores', required=True)
parser.add_argument('--top_k', type=int, default=3)
parser.add_argument('--output', default=None, help='default: next to --scores')
cli = parser.parse_args()

scores = json.load(open(cli.scores))
output = cli.output or cli.scores.replace('.json', '_image_sensations.csv')
with open(output, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['ID', 'sensation', 'score', f'top_{cli.top_k}'])
    for image_id, per_sensation in scores.items():
        ranked = sorted(((v[NORMALIZED_AVG_LOGPROB], k) for k, v in per_sensation.items()), reverse=True)
        top = ranked[:cli.top_k]
        writer.writerow([image_id, top[0][1], f'{top[0][0]:.4f}',
                         '; '.join(f'{name} ({score:.3f})' for score, name in top)])
print(f'wrote {len(scores)} images to {output}')
