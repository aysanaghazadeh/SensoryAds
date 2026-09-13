#!/usr/bin/env python3
"""
Build manifest + markdown for a Google Form-style annotation study:
10 sets × 10 pairwise comparisons (from AR_DALLE3_vs_AgenticEditing_sample.csv).

Each item shows **both** images (DALL·E 3 vs Agentic editing) and asks **which is better**
on sensation fit, message sets, and persuasion—not separate ratings per image.

Path .../<category>/<leaf>/<filename>:
  sensation_context = "{leaf}/{filename} → {category} → {leaf}"

JSON keys match leaf/filename; messages[0] = action-oriented, messages[1] = reason-oriented.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def parse_row_paths(dalle3_abs: Path, dataset_root_name: str = "AR_ALL_DALLE3") -> dict:
    """Return category, leaf_index, filename, relative path under dataset root."""
    parts = dalle3_abs.parts
    try:
        idx = parts.index(dataset_root_name)
    except ValueError:
        raise ValueError(f"{dataset_root_name} not in path: {dalle3_abs}")
    rel_parts = parts[idx + 1 :]
    if len(rel_parts) < 3:
        raise ValueError(f"need category/leaf/file under {dataset_root_name}: {dalle3_abs}")
    category = rel_parts[0]
    leaf = rel_parts[1]
    filename = rel_parts[-1]
    rel_under_root = "/".join(rel_parts)
    return {
        "category": category,
        "leaf_index": leaf,
        "filename": filename,
        "relative_under_ar": rel_under_root,
        "sensation_arrow": f"{leaf}/{filename} → {category} → {leaf}",
        "sensation_sentence": (
            f'Consider the sensation associated with category "{category}", '
            f'variant folder "{leaf}", image "{filename}" '
            f'(path key: {leaf}/{filename}).'
        ),
    }


def chunk(lst: list, n: int):
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--csv",
        type=Path,
        default=Path(__file__).resolve().parents[1]
        / "AR_DALLE3_vs_AgenticEditing_sample.csv",
    )
    ap.add_argument(
        "--subset-json",
        type=Path,
        default=Path(__file__).resolve().parents[1]
        / "QA_Combined_Action_Reason_AR_sample_subset.json",
        help="Preferred QA source; keys missing here are filled from --full-json.",
    )
    ap.add_argument(
        "--full-json",
        type=Path,
        default=Path(
            "/Users/aysanaghazadeh/University/Pitt/Research/Adriana/Data/PittAd/train/"
            "QA_Combined_Action_Reason_human_annotation_set.json"
        ),
        help="Fallback QA lookup for any image_key not in subset-json.",
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "annotation_google_form",
    )
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    with open(args.subset_json, encoding="utf-8") as f:
        qa_subset = json.load(f)

    qa_full = {}
    if args.full_json.is_file():
        with open(args.full_json, encoding="utf-8") as f:
            qa_full = json.load(f)

    def lookup_qa(key: str) -> list:
        if key in qa_subset:
            return qa_subset[key]
        if key in qa_full:
            return qa_full[key]
        raise KeyError(
            f"missing QA for key {key!r} (not in subset or full annotation file)"
        )

    rows_in = []
    with open(args.csv, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            dalle = Path(row["dalle3_path"])
            agentic = Path(row["agentic_editing_path"])
            meta = parse_row_paths(dalle)
            key = f'{meta["leaf_index"]}/{meta["filename"]}'
            pair = lookup_qa(key)
            action_msgs = pair[0]
            reason_msgs = pair[1]
            rows_in.append(
                {
                    "image_key": key,
                    "paths": {
                        "dalle3_absolute": str(dalle),
                        "agentic_absolute": str(agentic),
                        "forms_dalle3_relative": f"DALLE3/{meta['relative_under_ar']}",
                        "forms_agentic_relative": f"AgenticEditing/{meta['relative_under_ar']}",
                        # Optional: fill for google_apps_script/AnnotationFormBuilder.gs (Drive or HTTPS).
                        "drive_file_id_dalle3": None,
                        "drive_file_id_agentic": None,
                        "public_url_dalle3": None,
                        "public_url_agentic": None,
                        "comparison_labels": {
                            "A": "DALL·E 3 (left / first image)",
                            "B": "Agentic editing (right / second image)",
                        },
                    },
                    "sensation": {
                        "category": meta["category"],
                        "leaf_index": meta["leaf_index"],
                        "filename": meta["filename"],
                        "arrow_notation": meta["sensation_arrow"],
                        "instruction_fragment": meta["sensation_sentence"],
                    },
                    "messages": {
                        "action_or_primary": action_msgs,
                        "reason_or_secondary": reason_msgs,
                    },
                }
            )

    if len(rows_in) != 100:
        raise SystemExit(f"expected 100 CSV rows, got {len(rows_in)}")

    sets_of_10 = list(chunk(rows_in, 10))
    if len(sets_of_10) != 10:
        raise SystemExit(f"expected 10 sets, got {len(sets_of_10)}")

    comparison_options = [
        "A is clearly better",
        "A is somewhat better",
        "About the same / hard to choose",
        "B is somewhat better",
        "B is clearly better",
    ]

    manifest = {
        "layout": "paired_comparison",
        "instructions": (
            "For each pair, display **both** images side by side. "
            "Use fixed labels: **Image A** = DALL·E 3, **Image B** = Agentic editing. "
            "Ask which image is better on each criterion (not separate scores per image)."
        ),
        "comparison_options": comparison_options,
        "comparison_dimensions": [
            {
                "id": "evoke_sensation",
                "prompt": (
                    "Which image better evokes the intended sensation for this context? "
                    "(Context: see sensation path below.)"
                ),
            },
            {
                "id": "convey_action_messages",
                "prompt": (
                    "Which image better conveys or matches the **action-oriented messages** "
                    "(first message list below)?"
                ),
            },
            {
                "id": "convey_reason_messages",
                "prompt": (
                    "Which image better conveys or matches the **reason-oriented messages** "
                    "(second message list below)?"
                ),
            },
            {
                "id": "persuasive",
                "prompt": (
                    "Which image is **more persuasive** overall (for the advertised product or idea)?"
                ),
            },
        ],
        "sets": [],
    }

    for si, group in enumerate(sets_of_10, start=1):
        manifest["sets"].append(
            {
                "set_index": si,
                "pairs": [
                    {
                        "pair_index_in_set": ii + 1,
                        **item,
                    }
                    for ii, item in enumerate(group)
                ],
            }
        )

    manifest_path = args.out_dir / "google_form_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    # Markdown helper for human paste into Google Forms
    md_path = args.out_dir / "GOOGLE_FORM_QUESTIONS.md"
    lines = []
    lines.append("# Annotation study — pairwise comparison (paste into Google Forms)")
    lines.append("")
    lines.append(manifest["instructions"])
    lines.append("")
    lines.append(
        "Use **10 sections** (one per set). For each pair: add **two image items** on the same page "
        "(or one row in a table): **A** = DALL·E 3, **B** = Agentic editing. Upload from "
        "`~/experiments/forms/` using paths below."
    )
    lines.append("")
    lines.append("**Answer choices** (multiple choice, same for every comparison question):")
    for opt in manifest["comparison_options"]:
        lines.append(f"- {opt}")
    lines.append("")
    lines.append("---")
    for s in manifest["sets"]:
        lines.append("")
        lines.append(f"## Section {s['set_index']} — Set {s['set_index']} (10 pairs)")
        lines.append("")
        for img in s["pairs"]:
            lab = f"S{s['set_index']}-P{img['pair_index_in_set']}"
            lines.append(f"### {lab} — `{img['image_key']}`")
            lines.append("")
            lines.append(f"- **Sensation context:** {img['sensation']['arrow_notation']}")
            lines.append(f"- **Category:** {img['sensation']['category']} · **Variant folder:** {img['sensation']['leaf_index']} · **File:** {img['sensation']['filename']}")
            lines.append("- **Place images side by side:**")
            lines.append(f"  - **Image A (DALL·E 3):** `{img['paths']['forms_dalle3_relative']}`")
            lines.append(f"  - **Image B (Agentic editing):** `{img['paths']['forms_agentic_relative']}`")
            lines.append("")
            lines.append("**Action-oriented messages (set 1):**")
            for m in img["messages"]["action_or_primary"]:
                lines.append(f"- {m}")
            lines.append("")
            lines.append("**Reason-oriented messages (set 2):**")
            for m in img["messages"]["reason_or_secondary"]:
                lines.append(f"- {m}")
            lines.append("")
            lines.append(f"#### {lab} — comparison questions (choose A vs B)")
            for dim in manifest["comparison_dimensions"]:
                lines.append(f"- **{dim['id']}:** {dim['prompt']}")
            lines.append("")
            lines.append("---")

    md_path.write_text("\n".join(lines), encoding="utf-8")

    readme = args.out_dir / "README.txt"
    readme.write_text(
        "Generated by scripts/build_annotation_google_form.py\n\n"
        f"- {manifest_path.name}: machine-readable spec (10 sets × 10 images).\n"
        f"- {md_path.name}: question text for manual Google Form assembly.\n\n"
        "Pairwise layout: show A (DALL·E 3) and B (Agentic) together; each criterion is a single "
        "multiple-choice \"which is better\" question. Upload images from ~/experiments/forms/ "
        "using paths in the manifest.\n",
        encoding="utf-8",
    )

    print("Wrote:", manifest_path)
    print("Wrote:", md_path)
    print("Wrote:", readme)


if __name__ == "__main__":
    main()
