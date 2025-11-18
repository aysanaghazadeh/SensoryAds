python evaluation.py --config_type=DEFAULT \
--evaluation_type=Evosense_GT_Sensation \
--description_file=../experiments/results/SensoryAds/IN_QWenVL_20250918_122527_Sensation_QWenImage_Sensation_description_generation.csv \
--LLM=LLAMA3_instruct \
--fine_tuned=True \
--model_checkpoint=21000 \
--resume=True \
--AD_type=Sensation

python evaluate.py --config_type=DEFAULT \
--evaluation_type=llm_multi_question_persuasiveness_ranking \
--result_file=IN_InternVL_20250916_220717_AR_ALL_AuraFlow_ALL_description_generation.csv \
--VLM=InternVL \
--description_file=../experiments/results/SensoryAds/IN_InternVL_20250916_220717_AR_ALL_AuraFlow_ALL_description_generation.csv \
--LLM=LLAMA3_instruct

python evaluate.py --config_type=DEFAULT --evaluation_type=text_image_alignment \
--description_file=../experiments/results/SensoryAds/IN_InternVL_20250916_220717_AR_ALL_AuraFlow_ALL_description_generation.csv \
--LLM=LLAMA3_instruct \
--result_file=SensoryAds/IN_InternVL_20250916_220717_AR_ALL_AuraFlow_ALL_description_generation.csv \
--fine_tuned=True