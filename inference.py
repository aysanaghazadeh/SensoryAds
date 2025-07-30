from configs.inference_config import get_args
from utils.annotation.sensation_retreival import process_images
from utils.data.trian_test_split import get_test_data
if __name__ == "__main__":
    args = get_args()
    if args.inference_type == 'sensation_extraction':
        image_list = get_test_data(args)
        process_images(args, image_list)
        
    else:
        raise ValueError(f"Invalid inference type: {args.inference_type}")