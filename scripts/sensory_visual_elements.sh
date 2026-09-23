python -m SensoryVisualElements.pipeline \
--image_root=../experiments/generated_images/SensoryAds \
--output_dir=../experiments/results/SensoryAds/sensory_visual_elements \
--model_type=MLLM \
--MLLM=GPT4_o \
--LLM=GPT4o \
--canonicalization_batch_size=40 \
--resume=True
