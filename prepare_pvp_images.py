"""List the PVP images so inference.py's description_generation (Image_type=real, AD_type=ALL) can run on them.

That flow reads image IDs from the `ID` column of <data_path>/<test_set_QA> and opens
<data_path>/<test_set_images>/<ID>. The images live in <pvp_root>/image_files/<message>/<n>.jpg, so this
walks that folder (no Hugging Face access needed) and writes <pvp_root>/PVP_images.csv with one ID per image,
relative to image_files.

  python prepare_pvp_images.py --pvp_root=/ix/akovashka/aya34/projects/Data/PVP
"""
import argparse
import os

import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument('--pvp_root', required=True)
parser.add_argument('--images_dir', default='image_files')
parser.add_argument('--id_file', default='PVP_images.csv')
args = parser.parse_args()

image_root = os.path.join(args.pvp_root, args.images_dir)
ids = sorted(os.path.relpath(os.path.join(folder, name), image_root).replace(os.sep, '/')
             for folder, _, names in os.walk(image_root)
             for name in names if name.lower().endswith(('.jpg', '.jpeg', '.png')))

# description_generation stores ID as the last 3 path components, which only matches these IDs if they have at most 3
if any(i.count('/') > 2 for i in ids):
    print('WARNING: some IDs have more than 3 components; the description CSV IDs will be truncated.')

id_file = os.path.join(args.pvp_root, args.id_file)
pd.DataFrame({'ID': ids}).to_csv(id_file, index=False)
print(f'wrote {len(ids)} ids to {id_file}, e.g. {ids[:3]}')
