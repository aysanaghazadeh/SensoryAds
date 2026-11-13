
python evaluation.py --config_type=DEFAULT \
--evaluation_type=Evosense_GT_Sensation \
--description_file=../experiments/results/SensoryAds/IN_InternVL_ablation_20261110_105032_prio_none_AR_ALL_AgenticEditing_ALL_description_generation.csv \
--LLM=LLAMA3_instruct \
--fine_tuned=True \
--model_checkpoint=16500 \
--model_name=my_HierarchicalCPO_extended_annotation_LLAMA3_instruct \
--T2I_model=AuraFlow \
--test_set_sensation=train/sensation_annotations_parsed.json \
--AD_type=ALL \
--resume=True

python evaluation.py --config_type=DEFAULT \
--evaluation_type=Evosense_GT_Sensation \
--description_file=../experiments/results/SensoryAds/IN_InternVL_ablation_20261109_202423_no_text_refiner_AR_ALL_AgenticEditing_ALL_description_generation.csv \
--LLM=LLAMA3_instruct \
--fine_tuned=True \
--model_checkpoint=16500 \
--model_name=my_HierarchicalCPO_extended_annotation_LLAMA3_instruct \
--T2I_model=AuraFlow \
--test_set_sensation=train/sensation_annotations_parsed.json \
--AD_type=ALL \
--resume=True