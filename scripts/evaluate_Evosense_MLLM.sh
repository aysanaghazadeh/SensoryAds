# Scores every sensation for each test image with the MLLM trained by
# scripts/HierarchicalCPOTrainingMLLM.sh: log p(sensation | image), using the training prompt.
# MLLM is QWenVL or InternVL and must match the training run; MODEL_CHECKPOINT is a saved step
# under models/my_HierarchicalCPO_<MLLM>.
PYTHON=${PYTHON:-python}
MLLM=${MLLM:-InternVL}

$PYTHON evaluation.py --config_type=DEFAULT \
--evaluation_type=Evosense_MLLM \
--MLLM=$MLLM \
--fine_tuned=True \
--model_name=my_HierarchicalCPO_$MLLM \
--model_checkpoint=500 \
--sensation_annotations=train/sensation_annotations_parsed_1200_test.json \
--description_file=../experiments/results/SensoryAds/IN_InternVL_sensation_annotations_parsed_1200_test_ALL_real_description_generation.csv \
--resume=True
