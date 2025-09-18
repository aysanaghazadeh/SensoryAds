python inference.py --config_type=DEFAULT \
                    --inference_type=sensation_extraction \
                    --retrieval_type=multichoice \
                    --MLLM_prompt=MLLM_Sensation_Retrieval_Multichoice.jinja \
                    --model_type=MLLM \
                    --MLLM=LLAVA16 \
                    --resume=True