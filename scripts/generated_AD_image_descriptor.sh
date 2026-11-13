python inference.py \
--config_type=DEFAULT \
--AD_type=ALL \
--Image_type=generated \
--inference_type=description_generation \
--MLLM_prompt=description_generation.jinja \
--model_type=MLLM \
--MLLM=InternVL \
--description_goal=image_descriptor \
--test_set_images=../experiments/generated_images/SensoryAds/ablation_20261110_105032_prio_none/AR_ALL_AgenticEditing \
--test_set_QA=../experiments/results/SensoryAds/AR_ALL_AgenticEditing_ablation_20261110_105032_no_text_refiner.csv \
--resume=True

