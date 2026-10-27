"""Make the PVP images usable by inference.py's description_generation (Image_type=real, AD_type=ALL).

That flow reads image IDs from the `ID` column of <data_path>/<test_set_QA> and opens
<data_path>/<test_set_images>/<ID>, so this downloads the PVP repo files into
<data_path>/PVP_images and writes <data_path>/train/PVP_images.csv with one row per unique Image_Path.

  python prepare_pvp_images.py            # needs `huggingface-cli login` / HF_TOKEN and DATA_PATH set
"""
import argparse
import os

import pandas as pd
from datasets import load_dataset
from huggingface_hub import snapshot_download

parser = argparse.ArgumentParser()
parser.add_argument('--data_path', default=os.getenv('DATA_PATH'))
parser.add_argument('--images_dir', default='PVP_images')
parser.add_argument('--id_file', default='train/PVP_images.csv')
args = parser.parse_args()

ds = load_dataset('holi-lab/PVP')
pvp = pd.concat([ds[split].to_pandas() for split in ds], ignore_index=True)
ids = pvp.Image_Path.dropna().drop_duplicates()

image_root = os.path.join(args.data_path, args.images_dir)
snapshot_download('holi-lab/PVP', repo_type='dataset', local_dir=image_root)

found = ids.map(lambda p: os.path.exists(os.path.join(image_root, p)))
print(f'{found.sum()}/{len(ids)} Image_Paths resolve to a file under {image_root}')
if not found.all():
    print('first unresolved:', ids[~found].head(5).tolist())
    print('Fix the layout (or Image_Path prefix) before running description generation.')

# inference.py stores ID as the last 3 path components, so longer paths would no longer match Image_Path
if (ids.str.count('/') > 2).any():
    print('WARNING: some Image_Paths have more than 3 components; the description CSV IDs will be truncated.')

id_file = os.path.join(args.data_path, args.id_file)
pd.DataFrame({'ID': ids}).to_csv(id_file, index=False)
print(f'wrote {len(ids)} ids to {id_file}')
