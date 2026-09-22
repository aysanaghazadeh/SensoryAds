python evaluation.py --config_type=DEFAULT \
--evaluation_type=LLM \
--LLM=LLAMA3_instruct \
--model_type=LLM \
--LLM_prompt=LLM_judge.jinja \
--description_file=../experiments/results/SensoryAds/IN_InternVL_sensation_annotations_parsed_1200_test_ALL_real_description_generation.csv \
--resume=True
