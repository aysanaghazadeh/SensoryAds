python evaluation.py --config_type=DEFAULT \
--evaluation_type=Evosense_GT_Sensation \
--description_file=../experiments/results/SensoryAds/IN_InternVL_20260212_004312_AR_ALL_DALLE3_ALL_description_generation.csv \
--LLM=LLAMA3_instruct \
--fine_tuned=True \
--model_checkpoint=16500 \
--model_name=my_HierarchicalCPO_extended_annotation_LLAMA3_instruct \
--test_set_sensation=train/sensation_annotations_parsed.json \
--resume=True \
--AD_type=ALL



python evaluation.py --config_type=DEFAULT \
--evaluation_type=Evosense_GT_Sensation \
--description_file=../experiments/results/SensoryAds/IN_InternVL_20260129_002256_AR_ALL_AgenticEditing_ALL_description_generation.csv \
--LLM=LLAMA3_instruct \
--fine_tuned=True \
--model_checkpoint=16500 \
--model_name=my_HierarchicalCPO_extended_annotation_LLAMA3_instruct \
--T2I_model=AgenticEditing \
--AD_type=ALL \
--test_set_sensation=train/sensation_annotations_parsed.json \
--resume=True

python evaluation.py --config_type=DEFAULT \
--evaluation_type=Evosense_GT_Sensation \
--description_file=../experiments/results/SensoryAds/IN_InternVL_20250916_122348_AR_ALL_Flux_ALL_description_generation.csv \
--LLM=LLAMA3_instruct \
--fine_tuned=True \
--model_checkpoint=16500 \
--model_name=my_HierarchicalCPO_extended_annotation_LLAMA3_instruct \
--T2I_model=Flux \
--test_set_sensation=train/sensation_annotations_parsed.json \
--AD_type=ALL \
--resume=True

python evaluation.py --config_type=DEFAULT \
--evaluation_type=Evosense_GT_Sensation \
--description_file=../experiments/results/SensoryAds/IN_InternVL_20250916_130149_AR_ALL_SD3_ALL_description_generation.csv \
--LLM=LLAMA3_instruct \
--fine_tuned=True \
--model_checkpoint=16500 \
--model_name=my_HierarchicalCPO_extended_annotation_LLAMA3_instruct \
--T2I_model=SD3 \
--test_set_sensation=train/sensation_annotations_parsed.json \
--AD_type=ALL \
--resume=True

python evaluation.py --config_type=DEFAULT \
--evaluation_type=Evosense_GT_Sensation \
--description_file=../experiments/results/SensoryAds/IN_InternVL_20260224_024108_AR_ALL_QWenImage_ALL_description_generation.csv \
--LLM=LLAMA3_instruct \
--fine_tuned=True \
--model_checkpoint=16500 \
--model_name=my_HierarchicalCPO_extended_annotation_LLAMA3_instruct \
--T2I_model=QWenImage \
--test_set_sensation=train/sensation_annotations_parsed.json \
--AD_type=ALL \
--resume=True

python evaluation.py --config_type=DEFAULT \
--evaluation_type=Evosense_GT_Sensation \
--description_file=../experiments/results/SensoryAds/IN_InternVL_20250918_122434_AR_ALL_PixArt_ALL_description_generation.csv \
--LLM=LLAMA3_instruct \
--fine_tuned=True \
--model_checkpoint=16500 \
--model_name=my_HierarchicalCPO_extended_annotation_LLAMA3_instruct \
--T2I_model=PixArt \
--test_set_sensation=train/sensation_annotations_parsed.json \
--AD_type=ALL \
--resume=True

python evaluation.py --config_type=DEFAULT \
--evaluation_type=Evosense_GT_Sensation \
--description_file=../experiments/results/SensoryAds/IN_InternVL_20250916_220717_AR_ALL_AuraFlow_ALL_description_generation.csv \
--LLM=LLAMA3_instruct \
--fine_tuned=True \
--model_checkpoint=16500 \
--model_name=my_HierarchicalCPO_extended_annotation_LLAMA3_instruct \
--T2I_model=AuraFlow \
--test_set_sensation=train/sensation_annotations_parsed.json \
--AD_type=ALL \
--resume=True