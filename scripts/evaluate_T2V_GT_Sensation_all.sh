# T2V metrics (VQAScore, ImageReward, PickScore, CLIPScore). Needs the env with t2v_metrics (transformers<5).
# Run from the repo root: bash scripts/evaluate_T2V_GT_Sensation_all.sh
PYTHON=${PYTHON:-python}
RESULTS=../experiments/results/SensoryAds

# description csv (without .csv) | test_set_images
RUNS=(
"IN_InternVL_20260212_004312_AR_ALL_DALLE3_ALL_description_generation|20260212_004312/AR_ALL_DALLE3"
"IN_InternVL_20260129_002256_AR_ALL_AgenticEditing_ALL_description_generation|20260129_002256/AR_ALL_AgenticEditing"
"IN_InternVL_ablation_20261110_105032_prio_none_AR_ALL_AgenticEditing_ALL_description_generation|ablation_20261110_105032_prio_none/AR_ALL_AgenticEditing"
"IN_InternVL_20250916_122348_AR_ALL_Flux_ALL_description_generation|20250916_122348/AR_ALL_Flux"
)

for METRIC in VQA_score Image_Reward PickScore CLIPScore; do
  for run in "${RUNS[@]}"; do
    NAME="${run%%|*}"
    IMAGES="${run##*|}"
    $PYTHON evaluation.py --config_type=DEFAULT \
    --evaluation_type=${METRIC}_GT_Sensation \
    --description_file=$RESULTS/$NAME.csv \
    --test_set_images=$IMAGES \
    --Image_type=generated
  done
done
