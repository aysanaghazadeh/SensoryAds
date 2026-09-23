python train.py --config_type=DEFAULT \
--training_type=SFT_train_LLM \
--batch_size=8 \
--LLM=LLAMA3_instruct \
--sensation_annotations=train/sensation_annotations_parsed_1200_train.json \
--description_file=../experiments/results/SensoryAds/IN_InternVL_sensation_annotations_parsed_1200_train_ALL_real_description_generation.csv \
--train_set_QA=train/sensation_annotations_parsed_1200_train.csv 


