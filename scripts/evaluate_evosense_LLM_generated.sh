python evaluation.py --config_type=DEFAULT \
--evaluation_type=Evosense_LLM_generated \
--description_file=../experiments/results/SensoryAds/IN_QWenVL_20250917_185403_AR_ALL_QWenImage_ALL_description_generation.csv \
--MLLM=QWenVL \
--LLM=LLAMA3_instruct \
--fine_tuned=True \
--model_checkpoint=21000 \
--T2I_model=QWenImage \
--resume=True

python evaluation.py --config_type=DEFAULT \
--evaluation_type=Evosense_LLM_generated \
--description_file=../experiments/results/SensoryAds/IN_QWenVL_20250916_122348_AR_ALL_Flux_ALL_description_generation.csv \
--MLLM=QWenVL \
--LLM=QWenLM \
--fine_tuned=True \
--model_checkpoint=17500 \
--T2I_model=FLUX \
--resume=True

python evaluation.py --config_type=DEFAULT \
--evaluation_type=Evosense_LLM_generated \
--description_file=../experiments/results/SensoryAds/IN_InternVL_20250917_185403_AR_ALL_QWenImage_ALL_description_generation.csv \
--MLLM=InternVL \
--LLM=QWenLM \
--fine_tuned=True \
--model_checkpoint=17500 \
--T2I_model=QWenImage \
--resume=True


python evaluation.py --config_type=DEFAULT \
--evaluation_type=Evosense_LLM_generated \
--description_file=../experiments/results/SensoryAds/IN_InternVL_20250916_122348_AR_ALL_Flux_ALL_description_generation.csv \
--MLLM=InternVL \
--LLM=LLAMA3_instruct \
--T2I_model=FLUX \
--resume=True

python evaluation.py --config_type=DEFAULT \
--evaluation_type=Evosense_LLM_generated \
--description_file=../experiments/results/SensoryAds/IN_InternVL_20250917_185403_AR_ALL_QWenImage_ALL_description_generation.csv \
--MLLM=InternVL \
--LLM=QWenLM \
--T2I_model=QWenImage \
--resume=True
