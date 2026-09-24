# Each block is pinned to its own GPU and backgrounded so all three run at
# once instead of one after another; SLURM renumbers whatever GPUs it grants
# this job to 0..N-1, so index 0/1/2 here always lands on distinct GPUs.
pids=()

CUDA_VISIBLE_DEVICES=0 python evaluate.py --config_type=DEFAULT \
--evaluation_type=llm_multi_question_persuasiveness_ranking \
--result_file=SensoryAds/IN_InternVL_20260916_201638_AR_ALL_SD3_ALL_description_generation.csv  \
--VLM=InternVL \
--description_file=../experiments/results/SensoryAds/IN_InternVL_20260916_201638_AR_ALL_SD3_ALL_description_generation.csv \
--LLM=LLAMA3_instruct &
pids+=($!)

CUDA_VISIBLE_DEVICES=1 python evaluate.py --config_type=DEFAULT \
--evaluation_type=llm_multi_question_persuasiveness_ranking \
--result_file=SensoryAds/IN_InternVL_20260922_111803_AR_ALL_QWenImage_ALL_description_generation.csv  \
--VLM=InternVL \
--description_file=../experiments/results/SensoryAds/IN_InternVL_20260922_111803_AR_ALL_QWenImage_ALL_AuraFlow_ALL_description_generation.csv \
--LLM=LLAMA3_instruct &
pids+=($!)

CUDA_VISIBLE_DEVICES=2 python evaluate.py --config_type=DEFAULT \
--evaluation_type=llm_multi_question_persuasiveness_ranking \
--result_file=SensoryAds/IN_InternVL_20260908_001428_AR_ALL_PixArt_ALL_description_generation.csv  \
--VLM=InternVL \
--description_file=../experiments/results/SensoryAds/IN_InternVL_20260908_001428_AR_ALL_PixArt_ALL_description_generation.csv \
--LLM=LLAMA3_instruct &
pids+=($!)

# Wait for all three, but keep checking every PID (not the first failure)
# so one crashing early doesn't cut the other two short.
status=0
for pid in "${pids[@]}"; do
    wait "$pid" || status=1
done
exit $status
