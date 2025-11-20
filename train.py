# from configs.training_config import get_args
from configs.T2I_training_config import get_args
from Training.CPO_training_LLM import train as CPO_training_LLM
from Training.Hierarchical_CPO_training_LLM import train as HierarchicalCPO_training_LLM
from Training.SD3_training import train as SD3_training
if __name__ == '__main__':
    args = get_args()
    SD3_training(args)
    # if args.training_type == 'CPO_train_LLM':
    #     CPO_training_LLM(args)
    # if args.training_type == 'HierarchicalCPO_train_LLM':
    #     HierarchicalCPO_training_LLM(args)
