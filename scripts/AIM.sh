cd ../CAP
python evaluate.py --config_type=DEFAULT \
--evaluation_type=text_image_alignment \
--description_file=../experiments/results/SensoryAds/IN_InternVL_20260916_201638_AR_ALL_SD3_ALL_description_generation.csv \
--LLM=LLAMA3_instruct \
--result_file=SensoryAds/IN_InternVL_20260916_201638_AR_ALL_SD3_ALL_description_generation.csv \
--fine_tuned=True
