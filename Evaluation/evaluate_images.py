from Evaluation.sensation_alignment_metrics import *


class SensationEvaluation:
    def __init__(self, args):
        self.args = args

        if self.args.evaluation_type == 'Evosense_LLM':
            from LLMs.LLM import LLM
            self.model = LLM(args)
        if self.args.evaluation_type == 'Evosense_MLLM':
            from MLLMs.MLLM import MLLM
            self.model = MLLM(args)
        if self.args.evaluation_type == 'VQA_score':
            import t2v_metrics
            self.model = t2v_metrics.VQAScore(model='clip-flant5-xxl')

    def evaluate_Evosense_LLM(self, args):
        descriptions = pd.read_csv(args.description_file)
