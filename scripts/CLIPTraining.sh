PYTHON=${PYTHON:-python}

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
--training_type=CLIP_train \
--batch_size=128 \
--data_path=$DATA_PATH \
--sensation_annotations=train/sensation_annotations_parsed_1200_train.json \
--train_set_QA=train/sensation_annotations_parsed_1200_train.csv \
--AD_type=ALL
