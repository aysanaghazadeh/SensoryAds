python evaluation.py --config_type=DEFAULT \
--evaluation_type=Evosense_GT_Sensation \
--description_file=../experiments/results/SensoryAds/IN_InternVL_20251123_225258_AR_ALL_SD3_ALL_description_generation.csv \
--LLM=LLAMA3_instruct \
--fine_tuned=True \
--model_checkpoint=21000 \
--resume=True \
--AD_type=ALL



python evaluation.py --config_type=DEFAULT \
--evaluation_type=Evosense_GT_Sensation \
--description_file=../experiments/results/SensoryAds/IN_InternVL_20260129_002256_AR_ALL_AgenticEditing_ALL_description_generation.csv \
--LLM=LLAMA3_instruct \
--fine_tuned=True \
--model_checkpoint=20000 \
--T2I_model=AgenticEditing \
--AD_type=ALL \
--resume=True

python evaluate.py --config_type=DEFAULT \
--evaluation_type=text_image_alignment \
--result_path=../experiments/results/SensoryAds \
--result_file=IN_InternVL_20250916_220717_AR_ALL_AuraFlow_ALL_description_generation.csv \
--description_file=../experiments/results/SensoryAds/IN_InternVL_20260129_002256_AR_ALL_AgenticEditing_ALL_description_generation.csv \
--LLM=LLAMA3_instruct \
--fine_tuned=True 

python evaluate.py --config_type=DEFAULT \
--evaluation_type=llm_multi_question_persuasiveness_ranking \
--result_path=../experiments/results/SensoryAds \
--result_file=IN_InternVL_20250916_220717_AR_ALL_AuraFlow_ALL_description_generation.csv \
--VLM=InternVL \
--description_file=../experiments/results/SensoryAds/IN_InternVL_20250916_220717_AR_ALL_AuraFlow_ALL_description_generation.csv \
--LLM=LLAMA3_instruct