from configs.inference_config import get_args
from utils.annotation.sensation_retreival import process_files
from utils.data.trian_test_split import get_test_data
from generation.image_generation import generate_images

if __name__ == "__main__":
    args = get_args()
    if args.inference_type == 'sensation_extraction':
        image_list = get_test_data(args)
        process_files(args, image_list)
    elif args.inference_type == 'image_generation':
        generate_images(args)
    else:
        raise ValueError(f"Invalid inference type: {args.inference_type}")