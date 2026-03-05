python evaluation.py --config_type=DEFAULT \
--evaluation_type=MLLM_generated \
--description_file=../experiments/results/SensoryAds/IN_InternVL_20250917_185403_AR_ALL_QWenImage_ALL_description_generation.csv \
--test_set_images=20250917_185403/AR_ALL_QWenImage \
--T2I_model=QWenImage \
--Image_type=generated \
--MLLM=InternVL \
--MLLM_prompt=MLLM_judge.jinja \
--resume=True

python evaluation.py \
--config_type=DEFAULT \
--evaluation_type=MLLM_generated \
--description_file=../experiments/results/SensoryAds/IN_InternVL_20251123_225258_AR_ALL_SD3_ALL_description_generation.csv \
--evaluation_type=MLLM_generated \
--test_set_images=20251123_225258/AR_ALL_SD3 \
--T2I_model=SD3 \
--Image_type=generated \
--MLLM=QWenVL \
--MLLM_prompt=MLLM_judge.jinja \
--resume=True