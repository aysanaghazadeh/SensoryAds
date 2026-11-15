#!/bin/bash
# Run all GT-sensation evaluations (VQAScore, ImageReward, PickScore, CLIPScore,
# LLM judge [QWenLM, InternLM], MLLM judge [QWenVL, InternVL]) on each generation result.
# Run from the repo root: bash scripts/evaluate_all_single_sensation.sh

RESULTS=../experiments/results/SensoryAds

# description csv (without .csv)  |  test_set_images
RUNS=(
"IN_InternVL_20260212_004312_AR_ALL_DALLE3_ALL_description_generation|20260212_004312/AR_ALL_DALLE3"
"IN_InternVL_20260129_002256_AR_ALL_AgenticEditing_ALL_description_generation|20260129_002256/AR_ALL_AgenticEditing"
"IN_InternVL_ablation_20261110_105032_prio_none_AR_ALL_AgenticEditing_ALL_description_generation|ablation_20261110_105032_prio_none/AR_ALL_AgenticEditing"
"IN_InternVL_20250916_122348_AR_ALL_Flux_ALL_description_generation|20250916_122348/AR_ALL_Flux"
)

for run in "${RUNS[@]}"; do
  NAME="${run%%|*}"
  IMAGES="${run##*|}"
  DESC="$RESULTS/$NAME.csv"
  echo "===== $NAME ====="

  # T2V metrics
  for METRIC in VQA_score Image_Reward PickScore CLIPScore; do
    python evaluation.py --config_type=DEFAULT \
    --evaluation_type=${METRIC}_GT_Sensation \
    --description_file=$DESC \
    --test_set_images=$IMAGES \
    --Image_type=generated
  done

  # LLM judge (on the generated description)
  for LLM in QWenLM InternLM; do
    python evaluation.py --config_type=DEFAULT \
    --evaluation_type=LLM_GT_Sensation \
    --description_file=$DESC \
    --test_set_images=$IMAGES \
    --Image_type=generated \
    --model_type=LLM \
    --LLM=$LLM \
    --LLM_prompt=LLM_judge.jinja \
    --resume=True
  done

  # MLLM judge (on the generated image)
  for MLLM in QWenVL InternVL; do
    python evaluation.py --config_type=DEFAULT \
    --evaluation_type=MLLM_GT_Sensation \
    --description_file=$DESC \
    --test_set_images=$IMAGES \
    --Image_type=generated \
    --MLLM=$MLLM \
    --MLLM_prompt=MLLM_judge.jinja \
    --resume=True
  done
done
