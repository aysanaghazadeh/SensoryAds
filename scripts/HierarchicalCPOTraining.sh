python train.py --config_type=DEFAULT \
--training_type=HierarchicalCPO_train_LLM \
--batch_size=1 \
--LLM=QWenLM \
--sensation_annotations=train/sensation_annotations_parsed.json \
--description_file=../experiments/results/SensoryAds/IN_InternVL_train_images_total_ALL_description_generation.csv
