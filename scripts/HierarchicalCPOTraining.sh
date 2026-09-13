python train.py --config_type=DEFAULT \
--training_type=HierarchicalCPO_train_LLM \
--batch_size=1 \
--LLM=LLAMA3_instruct \
--sensation_annotations=train/ExtendedRealAdAnnotationsClean_parsed.json \
--description_file=../experiments/results/SensoryAds/IN_InternVL_PittAd_train_images_ALL_description_generation.csv

