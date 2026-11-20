# The LLM writes an image description per (ad, sensation) from the action-reason statements,
# then the T2I model renders that description.
python inference.py --config_type=DEFAULT \
                    --inference_type=image_generation \
                    --text_input_type=LLM \
                    --LLM=LLAMA3_instruct \
                    --LLM_prompt=LLM_input_physical_sensation.jinja \
                    --T2I_prompt=LLM_with_sensation.jinja \
                    --T2I_model=Flux \
                    --AD_type=SensoryAdsAll \
                    --with_physical_sensation=True \
                    --test_set_sensation=train/sensation_annotations_parsed.json \
                    --test_set_QA=train/SensoryAd_Action_Reason.json
