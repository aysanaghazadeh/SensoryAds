from configs.training_config import get_args
from Training.CPO_training_LLM import train as CPO_training_LLM
if __name__ == '__main__':
    args = get_args()
    if args.training_type == 'CPO_train_LLM':
        CPO_training_LLM(args)
