PYTHON=${PYTHON:-python}
echo "using interpreter: $($PYTHON -c 'import sys; print(sys.executable)')"

NUM_GPUS=$($PYTHON -c "import torch; print(torch.cuda.device_count())")
echo "detected ${NUM_GPUS} GPU(s)"

if [ "$NUM_GPUS" -gt 1 ]; then
    LAUNCH_ARGS="--multi_gpu --num_processes=${NUM_GPUS}"
else
    LAUNCH_ARGS="--num_processes=1"
fi

# Invoked as a module, not via the `accelerate` console script: that script's
# shebang hardcodes the interpreter of whichever env installed it, which breaks
# when the env is copied between machines.
$PYTHON -m accelerate.commands.launch $LAUNCH_ARGS train.py --config_type=DEFAULT \
--training_type=HierarchicalCPO_train_LLM \
--batch_size=1 \
--LLM=LLAMA3_instruct \
--sensation_annotations=train/sensation_annotations_parsed_1200_train.json \
--description_file=../experiments/results/SensoryAds/IN_InternVL_sensation_annotations_parsed_1200_train_ALL_real_description_generation.csv \
--train_set_QA=train/sensation_annotations_parsed_1200_train.csv \
--AD_type=ALL
