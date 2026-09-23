import os
import json
from collections import defaultdict

from SensoryVisualElements.config import get_args
from SensoryVisualElements.extraction_agent import ObjectExtractionAgent
from SensoryVisualElements.canonicalization_agent import CanonicalizationAgent


def find_sensation_folders(image_root, sensations=None):
    entries = sorted(
        entry for entry in os.listdir(image_root)
        if os.path.isdir(os.path.join(image_root, entry))
    )
    if sensations:
        entries = [entry for entry in entries if entry in sensations]
    return entries


def find_images(sensation_dir, extensions):
    extensions = {ext.lower() for ext in extensions}
    images = []
    for dirpath, _, filenames in os.walk(sensation_dir):
        for filename in sorted(filenames):
            if os.path.splitext(filename)[-1].lower() in extensions:
                images.append(os.path.join(dirpath, filename))
    return images


def load_json(path, default):
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return default


def save_json(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def extract_sensation_objects(args, extractor, sensation, images, raw_path):
    """Agent 1 pass: runs the extraction agent over every image of one sensation,
    checkpointing to raw_path so a crash/interrupt does not lose earlier work."""
    raw_mentions = load_json(raw_path, {}) if args.resume else {}
    for i, image_path in enumerate(images):
        if image_path in raw_mentions:
            continue
        try:
            raw_mentions[image_path] = extractor.extract(image_path, sensation)
        except Exception as exc:
            print(f'[extraction failed] {image_path}: {exc}')
            raw_mentions[image_path] = []
        if (i + 1) % 10 == 0 or (i + 1) == len(images):
            save_json(raw_path, raw_mentions)
            print(f'[{sensation}] extracted {i + 1}/{len(images)} images')
    save_json(raw_path, raw_mentions)
    return raw_mentions


def pool_raw_counts(raw_mentions):
    """Collapse exact-string duplicates for free before handing anything to the
    (costlier) canonicalization agent, which only needs to resolve genuinely
    different-looking mentions of the same object."""
    counts = defaultdict(int)
    for objects in raw_mentions.values():
        for obj in objects:
            normalized = ' '.join(obj.strip().lower().split())
            if normalized:
                counts[normalized] += 1
    return dict(counts)


def run(args):
    extractor = ObjectExtractionAgent(args)
    canonicalizer = CanonicalizationAgent(args)

    sensations = find_sensation_folders(args.image_root, args.sensations)
    print(f'found {len(sensations)} sensation folders: {sensations}')

    summary = {}
    for sensation in sensations:
        sensation_dir = os.path.join(args.image_root, sensation)
        images = find_images(sensation_dir, args.image_extensions)
        if args.max_images_per_sensation:
            images = images[:args.max_images_per_sensation]
        print(f'[{sensation}] {len(images)} images found')

        raw_path = os.path.join(args.output_dir, 'raw', f'{sensation}.json')
        raw_mentions = extract_sensation_objects(args, extractor, sensation, images, raw_path)

        raw_counts = pool_raw_counts(raw_mentions)
        print(f'[{sensation}] {len(raw_counts)} unique raw mentions '
              f'from {sum(raw_counts.values())} total mentions')

        canonical_counts = canonicalizer.canonicalize(sensation, raw_counts, args.canonicalization_batch_size)
        canonical_counts = dict(sorted(canonical_counts.items(), key=lambda kv: -kv[1]))

        canonical_path = os.path.join(args.output_dir, f'{sensation}.json')
        save_json(canonical_path, canonical_counts)
        print(f'[{sensation}] {len(canonical_counts)} canonical objects -> {canonical_path}')

        summary[sensation] = canonical_counts

    summary_path = os.path.join(args.output_dir, 'summary.json')
    save_json(summary_path, summary)
    print(f'done. summary written to {summary_path}')


if __name__ == '__main__':
    run(get_args())
