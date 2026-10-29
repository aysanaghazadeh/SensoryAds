PYTHON=${PYTHON:-python}
MLLM=${MLLM:-InternVL}  # QWenVL or InternVL

# An empty value silently turns every data path into a path relative to the cwd.
: "${DATA_PATH:?set DATA_PATH to the PittAd folder that contains train/}"

NUM_GPUS=$($PYTHON -c "import torch; print(torch.cuda.device_count())")
echo "detected ${NUM_GPUS} GPU(s)"

# See SFTTraining.sh for why the port is derived from the job ID and why
# accelerate is invoked as a module.
if [ "$NUM_GPUS" -gt 1 ]; then
    MAIN_PROCESS_PORT=$(( 20000 + (${SLURM_JOB_ID:-$$} % 20000) ))
    LAUNCH_ARGS="--multi_gpu --num_processes=${NUM_GPUS} --main_process_ip=127.0.0.1 --main_process_port=${MAIN_PROCESS_PORT}"
else
    LAUNCH_ARGS="--num_processes=1"
fi

$PYTHON -m accelerate.commands.launch $LAUNCH_ARGS train.py --config_type=DEFAULT \
--training_type=HierarchicalCPO_train_MLLM \
--batch_size=4 \
--MLLM=$MLLM \
--data_path=$DATA_PATH \
--sensation_annotations=train/sensation_annotations_parsed_1200_train.json \
--train_set_QA=train/sensation_annotations_parsed_1200_train.csv \
--AD_type=ALL
