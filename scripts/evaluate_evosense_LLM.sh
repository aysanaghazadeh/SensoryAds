python evaluation.py --config_type=DEFAULT \
--evaluation_type=Evosense_LLM \
--description_file=../experiments/results/SensoryAds/IN_InternVL_sensation_annotations_parsed_1200_test_ALL_real_description_generation.csv \
--LLM=QWenLM \
--fine_tuned=True \
--model_checkpoint=12500 \
--model_name=my_HierarchicalCPO_extended_annotation_QWenLM \
--sensation_annotations=train/sensation_annotations_parsed_1200_test.json



