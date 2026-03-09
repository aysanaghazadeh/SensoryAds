python evaluation.py --config_type=DEFAULT \
--evaluation_type=LLM_generated \
--description_file=../experiments/results/SensoryAds/IN_InternVL_20250917_185403_AR_ALL_QWenImage_ALL_description_generation.csv \
--T2I_model=QWenImage \
--Image_type=generated \
--model_type=LLM \
--LLM=QWenLM \
--LLM_prompt=LLM_judge.jinja \
--resume=True

python evaluation.py \
--config_type=DEFAULT \
--evaluation_type=LLM_generated \
--description_file=../experiments/results/SensoryAds/IN_InternVL_20251123_225258_AR_ALL_SD3_ALL_description_generation.csv \
--T2I_model=SD3 \
--Image_type=generated \
--model_type=LLM \
--LLM=LLAMA3_instruct \
--LLM_prompt=LLM_judge.jinja \
--resume=True

python evaluation.py \
--config_type=DEFAULT \
--evaluation_type=LLM_generated \
--description_file=../experiments/results/SensoryAds/IN_InternVL_20250916_122348_AR_ALL_Flux_ALL_description_generation.csv \
--T2I_model=FLUX \
--Image_type=generated \
--model_type=LLM \
--LLM=QWenLM \
--LLM_prompt=LLM_judge.jinja \
--resume=True

python evaluation.py \
--config_type=DEFAULT \
--evaluation_type=LLM_generated \
--description_file=../experiments/results/SensoryAds/IN_InternVL_20250916_220717_AR_ALL_AuraFlow_ALL_description_generation.csv \
--T2I_model=AuraFlow \
--Image_type=generated \
--model_type=LLM \
--LLM=LLAMA3_instruct \
--LLM_prompt=LLM_judge.jinja \
--resume=True