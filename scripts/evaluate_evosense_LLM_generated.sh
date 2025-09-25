python evaluation.py --config_type=DEFAULT \
--evaluation_type=Evosense_LLM_generated \
--description_file=../experiments/results/SensoryAds/IN_InternVL_20250918_034952_Sensation_SD3_Sensation_description_generation.csv \
--LLM=LLAMA3_instruct \
--fine_tuned=True \
--model_checkpoint=21000 \
--T2I_model=SD3 \
--resume=True
