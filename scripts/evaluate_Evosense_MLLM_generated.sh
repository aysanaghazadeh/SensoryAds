# Evosense_MLLM on generated images: scores log p(sensation | image) with the MLLM trained by
# scripts/HierarchicalCPOTrainingMLLM.sh. Images are read from
# <result_path>/generated_images/<project_name>/<test_set_images>/<ID>, as in evaluate_CLIPScore_generated.sh.
# MLLM must match the training run; MODEL_CHECKPOINT is a saved step under models/my_HierarchicalCPO_<MLLM>.
PYTHON=${PYTHON:-python}
MLLM=${MLLM:-InternVL}
MODEL_CHECKPOINT=${MODEL_CHECKPOINT:-500}

$PYTHON evaluation.py --config_type=DEFAULT \
--evaluation_type=Evosense_MLLM \
--description_file=../experiments/results/SensoryAds/IN_InternVL_20250916_122348_AR_ALL_Flux_ALL_description_generation.csv \
--test_set_images=20250916_122348/AR_ALL_Flux \
--MLLM=$MLLM \
--fine_tuned=True \
--model_name=my_HierarchicalCPO_$MLLM \
--model_checkpoint=$MODEL_CHECKPOINT \
--T2I_model=FLUX \
--Image_type=generated \
--resume=True

$PYTHON evaluation.py --config_type=DEFAULT \
--evaluation_type=Evosense_MLLM \
--description_file=../experiments/results/SensoryAds/IN_InternVL_20251123_225258_AR_ALL_SD3_ALL_description_generation.csv \
--test_set_images=20251123_225258/AR_ALL_SD3 \
--MLLM=$MLLM \
--fine_tuned=True \
--model_name=my_HierarchicalCPO_$MLLM \
--model_checkpoint=$MODEL_CHECKPOINT \
--T2I_model=SD3 \
--Image_type=generated \
--resume=True

$PYTHON evaluation.py --config_type=DEFAULT \
--evaluation_type=Evosense_MLLM \
--description_file=../experiments/results/SensoryAds/IN_InternVL_20250917_185403_AR_ALL_QWenImage_ALL_description_generation.csv \
--test_set_images=20250917_185403/AR_ALL_QWenImage \
--MLLM=$MLLM \
--fine_tuned=True \
--model_name=my_HierarchicalCPO_$MLLM \
--model_checkpoint=$MODEL_CHECKPOINT \
--T2I_model=QWenImage \
--Image_type=generated \
--resume=True

$PYTHON evaluation.py --config_type=DEFAULT \
--evaluation_type=Evosense_MLLM \
--description_file=../experiments/results/SensoryAds/IN_InternVL_20260129_002256_AR_ALL_AgenticEditing_ALL_description_generation.csv \
--test_set_images=20260129_002256/AR_ALL_AgenticEditing \
--MLLM=$MLLM \
--fine_tuned=True \
--model_name=my_HierarchicalCPO_$MLLM \
--model_checkpoint=$MODEL_CHECKPOINT \
--T2I_model=AgenticEditing \
--Image_type=generated \
--resume=True
