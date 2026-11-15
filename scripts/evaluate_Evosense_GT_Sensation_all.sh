# Evosense (GT sensation) on the generated-image descriptions.
#   Evosense (HierarchicalCPO) QWenLM       : InternVL + QWenVL descriptions
#   Evosense (HierarchicalCPO) LLAMA3_instruct: QWenVL descriptions only
#   Zero-shot LLAMA3_instruct               : InternVL descriptions only
#   CPO LLAMA3_instruct                     : InternVL descriptions only
#   SFT LLAMA3_instruct                     : InternVL descriptions only
# Run from the repo root: bash scripts/evaluate_Evosense_GT_Sensation_all.sh
RESULTS=../experiments/results/SensoryAds

# Checkpoints inside ../models/<model_name>/checkpoint-<N>. CPO/SFT are the ones
# last used in evaluate_evosense_LLM.sh (git history); override with
# CPO_CKPT=... SFT_CKPT=... if needed.
EVOSENSE_QWEN_CKPT=11500
EVOSENSE_LLAMA_CKPT=16500
CPO_CKPT=${CPO_CKPT:-7000}
SFT_CKPT=${SFT_CKPT:-8000}

# InternVL description csvs (without .csv); the QWenVL ones swap InternVL -> QWenVL
INTERNVL_NAMES=(
"IN_InternVL_20260212_004312_AR_ALL_DALLE3_ALL_description_generation"
"IN_InternVL_20260129_002256_AR_ALL_AgenticEditing_ALL_description_generation"
"IN_InternVL_ablation_20261110_105032_prio_none_AR_ALL_AgenticEditing_ALL_description_generation"
"IN_InternVL_20250916_122348_AR_ALL_Flux_ALL_description_generation"
)

run_evosense() {  # description_name LLM [fine_tuned model_name checkpoint]
  local NAME=$1 LLM=$2 FT=${3:-False} MODEL_NAME=$4 CKPT=$5
  local ARGS=()
  if [ "$FT" = "True" ]; then
    ARGS=(--fine_tuned=True --model_name=$MODEL_NAME --model_checkpoint=$CKPT)
  fi
  python evaluation.py --config_type=DEFAULT \
  --evaluation_type=Evosense_GT_Sensation \
  --description_file=$RESULTS/$NAME.csv \
  --LLM=$LLM \
  "${ARGS[@]}" \
  --AD_type=ALL \
  --resume=True
}

for NAME in "${INTERNVL_NAMES[@]}"; do
  QWENVL_NAME=${NAME/InternVL/QWenVL}

  # Evosense QWenLM: both description sets
  run_evosense $NAME        QWenLM True my_HierarchicalCPO_extended_annotation_QWenLM $EVOSENSE_QWEN_CKPT
  run_evosense $QWENVL_NAME QWenLM True my_HierarchicalCPO_extended_annotation_QWenLM $EVOSENSE_QWEN_CKPT

  # Evosense LLAMA3_instruct: QWenVL descriptions only
  run_evosense $QWENVL_NAME LLAMA3_instruct True my_HierarchicalCPO_extended_annotation_LLAMA3_instruct $EVOSENSE_LLAMA_CKPT

  # InternVL descriptions only: zero-shot, CPO, SFT (LLAMA3_instruct)
  run_evosense $NAME LLAMA3_instruct False
  run_evosense $NAME LLAMA3_instruct True myCPO_LLAMA3_instruct $CPO_CKPT
  run_evosense $NAME LLAMA3_instruct True mySFT_LLAMA3_instruct $SFT_CKPT
done
