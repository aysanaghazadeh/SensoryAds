PYTHON=${PYTHON:-python}

NUM_GPUS=$($PYTHON -c "import torch; print(torch.cuda.device_count())")
echo "detected ${NUM_GPUS} GPU(s)"

# port=0 (dynamic) left the worker processes trying to connect on a literal
# port 0 and hanging forever, so derive a fixed port from the job ID instead
# - unique enough per job to avoid colliding with another job's rendezvous on
# a shared node. main_process_ip pins the loopback address explicitly so c10d
# never has to resolve "localhost" (which was falling back from an unusable
# IPv6 result with errno 97 on this cluster).
if [ "$NUM_GPUS" -gt 1 ]; then
    MAIN_PROCESS_PORT=$(( 20000 + (${SLURM_JOB_ID:-$$} % 20000) ))
    LAUNCH_ARGS="--multi_gpu --num_processes=${NUM_GPUS} --main_process_ip=127.0.0.1 --main_process_port=${MAIN_PROCESS_PORT}"
else
    LAUNCH_ARGS="--num_processes=1"
fi

# Invoked as a module, not via the `accelerate` console script: that script's
# shebang hardcodes the interpreter of whichever env installed it, which breaks
# when the env is copied between machines.
$PYTHON -m accelerate.commands.launch $LAUNCH_ARGS train.py --config_type=DEFAULT \
--training_type=CPO_train_LLM \
--batch_size=16 \
--LLM=LLAMA3_instruct \
--sensation_annotations=train/sensation_annotations_parsed_1200_train.json \
--description_file=../experiments/results/SensoryAds/IN_InternVL_sensation_annotations_parsed_1200_train_ALL_real_description_generation.csv \
--train_set_QA=train/sensation_annotations_parsed_1200_train.csv \
--model_checkpoint=1000

