python evaluation.py --config_type=DEFAULT \
--evaluation_type=Evosense_LLM \
--description_file=../experiments/results/SensoryAds/IN_InternVL_PVP_images_ALL_real_description_generation.csv \
--LLM=LLAMA3_instruct \
--fine_tuned=True \
--model_checkpoint=16500 \
--model_name=my_HierarchicalCPO_extended_annotation_LLAMA3_instruct \
--resume=True
