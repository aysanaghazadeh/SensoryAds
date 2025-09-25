python evaluation.py --config_type=DEFAULT \
--evaluation_type=Evosense_LLM_generated \
--description_file=../experiments/results/SensoryAds/IN_InternVL_20250916_122348_AR_ALL_Flux_ALL_description_generation.csv \
--LLM=LLAMA3_instruct \
--fine_tuned=True \
--model_checkpoint=21000 \
--T2I_model=FLUX \
--resume=True
