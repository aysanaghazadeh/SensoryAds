python evaluation.py --config_type=DEFAULT \
--evaluation_type=Evosense_LLM \
--description_file=../experiments/results/SensoryAds/IN_InternVL_train_images_total_ALL_description_generation.csv \
--LLM=LLAMA3_instruct \
--fine_tuned=True \
--model_checkpoint=21000 \
--model_name=my_LLAMA3_instruct \
--resume=True
