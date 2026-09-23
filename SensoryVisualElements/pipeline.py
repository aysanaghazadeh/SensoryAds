import os
import json
from collections import defaultdict

from SensoryVisualElements.config import get_args
from SensoryVisualElements.extraction_agent import ObjectExtractionAgent
from SensoryVisualElements.canonicalization_agent import CanonicalizationAgent


def find_images_by_sensation(image_root, extensions, sensations=None):
    """Walk image_root once and group every image by its sensation, read directly
    off the path: image_root / <sensation> / <...any subfolders...> / <image>."""
    extensions = {ext.lower() for ext in extensions}
    images_by_sensation = defaultdict(list)
    for dirpath, _, filenames in os.walk(image_root):
        rel_dir = os.path.relpath(dirpath, image_root)
        if rel_dir == '.':
            continue  # images directly under image_root belong to no sensation
        sensation = rel_dir.split(os.sep)[0]
        if sensations and sensation not in sensations:
            continue
        for filename in sorted(filenames):
            if os.path.splitext(filename)[-1].lower() in extensions:
                images_by_sensation[sensation].append(os.path.join(dirpath, filename))
    return dict(images_by_sensation)


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

    images_by_sensation = find_images_by_sensation(args.image_root, args.image_extensions, args.sensations)
    if args.max_images_per_sensation:
        images_by_sensation = {
            sensation: images[:args.max_images_per_sensation]
            for sensation, images in images_by_sensation.items()
        }
    print(f'found {len(images_by_sensation)} sensations: {sorted(images_by_sensation)}')

    sensory_visual_elements = {}
    for sensation in sorted(images_by_sensation):
        images = images_by_sensation[sensation]
        print(f'[{sensation}] {len(images)} images found')

        raw_path = os.path.join(args.output_dir, 'raw', f'{sensation}.json')
        raw_mentions = extract_sensation_objects(args, extractor, sensation, images, raw_path)

        raw_counts = pool_raw_counts(raw_mentions)
        print(f'[{sensation}] {len(raw_counts)} unique raw mentions '
              f'from {sum(raw_counts.values())} total mentions')

        canonical_counts = canonicalizer.canonicalize(sensation, raw_counts, args.canonicalization_batch_size)
        canonical_counts = dict(sorted(canonical_counts.items(), key=lambda kv: -kv[1]))
        print(f'[{sensation}] {len(canonical_counts)} canonical objects')

        sensory_visual_elements[sensation] = canonical_counts

    output_path = os.path.join(args.output_dir, 'sensory_visual_elements.json')
    save_json(output_path, sensory_visual_elements)
    print(f'done. {{sensation: {{object: count}}}} map written to {output_path}')
    return sensory_visual_elements


if __name__ == '__main__':
    run(get_args())
