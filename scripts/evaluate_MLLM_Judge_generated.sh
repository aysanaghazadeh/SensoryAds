python evaluation.py --config_type=DEFAULT \
--evaluation_type=MLLM_generated \
--description_file=../experiments/results/SensoryAds/gen_images_human_annotated_images.csv \
--evaluation_type=MLLM_generated \
--MLLM=InternVL \
--MLLM_prompt=MLLM_judge.jinja \
--resume=True
