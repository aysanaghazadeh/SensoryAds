python evaluation.py --config_type=DEFAULT \
--evaluation_type=Evosense_GT_Sensation \
--description_file=../experiments/results/SensoryAds/IN_InternVL_20251123_225258_AR_ALL_SD3_ALL_description_generation.csv \
--LLM=LLAMA3_instruct \
--fine_tuned=True \
--model_checkpoint=40000 \
--evaluate_other_evoked_sensations=True \
--test_set_sensation=train/sensation_annotations_parsed.json \
--resume=True \
--AD_type=ALL



python evaluation.py --config_type=DEFAULT \
--evaluation_type=Evosense_GT_Sensation \
--description_file=../experiments/results/SensoryAds/IN_InternVL_20250917_185403_AR_ALL_QWenImage_ALL_description_generation.csv \
--LLM=LLAMA3_instruct \
--fine_tuned=True \
--model_checkpoint=40000 \
--model_name=my_HierarchicalCPO_data_annotation_LLAMA3_instruct \
--T2I_model=QwenImage \
--AD_type=ALL \
--evaluate_other_evoked_sensations=True \
--test_set_sensation=train/sensation_annotations_parsed.json \
--resume=True

python evaluation.py --config_type=DEFAULT \
--evaluation_type=Evosense_GT_Sensation \
--description_file=../experiments/results/SensoryAds/IN_InternVL_20260129_002256_AR_ALL_AgenticEditing_ALL_description_generation.csv \
--LLM=LLAMA3_instruct \
--fine_tuned=True \
--model_checkpoint=40000 \
--T2I_model=AgenticEditing \
--evaluate_other_evoked_sensations=True \
--test_set_sensation=train/sensation_annotations_parsed.json \
--AD_type=ALL \
--resume=True

python evaluation.py --config_type=DEFAULT \
--evaluation_type=Evosense_GT_Sensation \
--description_file=../experiments/results/SensoryAds/IN_InternVL_20250916_122348_AR_ALL_Flux_ALL_description_generation.csv \
--LLM=LLAMA3_instruct \
--fine_tuned=True \
--model_checkpoint=40000 \
--T2I_model=AgenticEditing \
--evaluate_other_evoked_sensations=True \
--test_set_sensation=train/sensation_annotations_parsed.json \
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