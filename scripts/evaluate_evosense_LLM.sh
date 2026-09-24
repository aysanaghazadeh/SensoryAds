python evaluation.py --config_type=DEFAULT \
--evaluation_type=Evosense_LLM \
--description_file=../experiments/results/SensoryAds/IN_QWenVL_sensation_annotations_parsed_1200_test_ALL_real_description_generation.csv \
--LLM=QWenLM \
--fine_tuned=True \
--model_checkpoint=11500 \
--model_name=my_HierarchicalCPO_extended_annotation_QWenLM \
--sensation_annotations=train/sensation_annotations_parsed_1200_test.json

python evaluation.py --config_type=DEFAULT \
--evaluation_type=Evosense_LLM \
--description_file=../experiments/results/SensoryAds/IN_InternVL_sensation_annotations_parsed_1200_test_ALL_real_description_generation.csv \
--LLM=LLAMA3_instruct \
--fine_tuned=True \
--model_checkpoint=7000 \
--model_name=myCPO_LLAMA3_instruct \
--sensation_annotations=train/sensation_annotations_parsed_1200_test.json


