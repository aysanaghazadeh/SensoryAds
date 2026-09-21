PYTHON=${PYTHON:-python}
# Relative data paths resolve against the launch directory, which differs
# between interactive shells and batch jobs; override DATA_PATH to pin it.
DATA_PATH=${DATA_PATH:-../Data/PittAd}
EXPERIMENTS_PATH=${EXPERIMENTS_PATH:-../experiments}

echo "using interpreter: $($PYTHON -c 'import sys; print(sys.executable)')"
echo "cwd: $(pwd)"
echo "data path: ${DATA_PATH} (exists: $([ -d "$DATA_PATH" ] && echo yes || echo NO))"

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
--batch_size=16 \
--LLM=LLAMA3_instruct \
--data_path=$DATA_PATH \
--sensation_annotations=train/sensation_annotations_parsed_1200_train.json \
--description_file=$EXPERIMENTS_PATH/results/SensoryAds/IN_InternVL_sensation_annotations_parsed_1200_train_ALL_real_description_generation.csv \
--train_set_QA=train/sensation_annotations_parsed_1200_train.csv \
--AD_type=ALL \
--model_checkpoint=4500
