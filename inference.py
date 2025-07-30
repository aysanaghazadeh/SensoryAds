from configs.inference_config import get_args
from analysis.attention_map import process_image
from reasoning.reasoning import reasoning
from util.data.data_loader import get_analysis_data

if __name__ == "__main__":
    args = get_args()
    if args.inference_type == 'analysis':
        image_list, bounding_box_list, prompts, image_ids = get_analysis_data(args)
        IoU_list, recall_list, precision_list = process_image(args, image_list, bounding_box_list, prompts, image_ids)
    elif args.inference_type == 'QA':
        reasoning(args)
    else:
        raise ValueError(f"Invalid inference type: {args.inference_type}")