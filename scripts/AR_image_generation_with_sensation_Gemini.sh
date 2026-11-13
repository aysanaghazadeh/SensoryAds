
python inference.py --config_type=DEFAULT \
                    --inference_type=image_generation \
                    --text_input_type=AR \
                    --T2I_model=Gemini \
                    --gemini_model=gemini-3.1-flash-image \
                    --T2I_prompt=AR_with_sensation.jinja \
                    --AD_type=SensoryAdsAll \
                    --with_physical_sensation=True \
                    --test_set_sensation=train/sensation_annotations_parsed.json \
                    --test_set_QA=train/SensoryAd_Action_Reason.json
