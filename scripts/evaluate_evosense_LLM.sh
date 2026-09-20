python evaluation.py --config_type=DEFAULT \
--evaluation_type=Evosense_LLM \
--description_file=../experiments/results/SensoryAds/IN_InternVL_sensation_annotations_parsed_1200_test_ALL_real_description_generation.csv \
--LLM=LLAMA3_instruct \
--fine_tuned=True \
--model_checkpoint=10500 \
--model_name=my_HierarchicalCPO_extended_annotation_LLAMA3_instruct \
--resume=True \
--sensation_annotations=train/sensation_annotations_parsed_1200_test.json

/experiments/results/SensoryAds/Evosense_LLM/IN_InternVL_train_images_total_ALL_description_generation_LLAMA3_instruct_finetunedTrue_my_HierarchicalCPO_LLAMA3_instruct64500.json