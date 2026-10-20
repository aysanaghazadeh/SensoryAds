# Scores with the SFT_score_train_LLM model. SFT_score_judge.jinja is the exact training
# prompt; the model answers with a bare 0-5 score, which get_LLM_Judge_Score divides by 5.
# MODEL_CHECKPOINT is the step of a saved checkpoint under models/mySFTScore_LLAMA3_instruct.
python evaluation.py --config_type=DEFAULT \
--evaluation_type=LLM \
--LLM=LLAMA3_instruct \
--model_type=LLM \
--LLM_prompt=SFT_score_judge.jinja \
--fine_tuned=True \
--model_name=mySFTScore_LLAMA3_instruct \
--model_checkpoint=${MODEL_CHECKPOINT:?set MODEL_CHECKPOINT to a saved checkpoint step} \
--project_name=SensoryAds_SFTScore \
--description_file=../experiments/results/SensoryAds/IN_InternVL_sensation_annotations_parsed_1200_test_ALL_real_description_generation.csv \
--resume=True
