import argparse
import os


def str2bool(value):
    if isinstance(value, bool):
        return value
    if value.lower() in ('true', 'yes', '1'):
        return True
    if value.lower() in ('false', 'no', '0'):
        return False
    raise argparse.ArgumentTypeError(f'invalid boolean value: {value}')


def get_args():
    parser = argparse.ArgumentParser(
        description='Extract and de-duplicate the sensory visual elements (objects) shown in a set of '
                     'sensation-labeled advertisement images.'
    )

    # data layout: image_root/<sensation>/<any subfolders>/<images>
    parser.add_argument('--image_root', type=str, required=True,
                         help='root folder containing one subfolder per sensation')
    parser.add_argument('--output_dir', type=str, required=True,
                         help='where per-sensation object dictionaries and raw extraction logs are written')
    parser.add_argument('--sensations', type=str, nargs='*', default=None,
                         help='restrict to these sensation subfolder names (default: all subfolders of image_root)')
    parser.add_argument('--image_extensions', type=str, nargs='*',
                         default=['.jpg', '.jpeg', '.png', '.webp', '.bmp'])
    parser.add_argument('--max_images_per_sensation', type=int, default=None,
                         help='cap the number of images processed per sensation, useful for a quick test run')

    # extraction agent (vision)
    parser.add_argument('--model_type', type=str, default='MLLM', choices=['MLLM'])
    parser.add_argument('--MLLM', type=str, default='GPT4_o')
    parser.add_argument('--extraction_prompt', type=str, default='sensory_object_extraction.jinja')
    parser.add_argument('--extraction_max_new_tokens', type=int, default=200,
                         help='generate_kwargs max_new_tokens passed to local-model MLLM backends '
                              '(ignored by GPT4_o/Gemini, whose forward() does not take generate_kwargs)')

    # canonicalization agent (text)
    parser.add_argument('--LLM', type=str, default='GPT4o')
    parser.add_argument('--canonicalization_prompt', type=str, default='sensory_object_canonicalization.jinja')
    parser.add_argument('--canonicalization_batch_size', type=int, default=40,
                         help='how many unique raw object mentions are resolved against the dictionary per LLM call')

    # shared prompt location, matches utils/prompt_engineering/prompts
    parser.add_argument('--prompt_path', type=str,
                         default=os.path.join('utils', 'prompt_engineering', 'prompts'))

    # local-model backends (QWenVL/InternVL/LLAMA3.../GPT4_o all read a subset of these)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--train', type=str2bool, default=False)
    parser.add_argument('--fine_tuned', type=str2bool, default=False)
    parser.add_argument('--model_path', type=str, default=None)
    parser.add_argument('--model_name', type=str, default=None)
    parser.add_argument('--model_checkpoint', type=str, default=None)
    parser.add_argument('--api_key', type=str, default=os.environ.get('OPENAI_API_KEY'),
                         help='defaults to the OPENAI_API_KEY already in the environment; '
                              'the LLM=GPT4o text backend requires this to be set one way or the other')

    parser.add_argument('--resume', type=str2bool, default=True,
                         help='skip images already present in the per-sensation raw extraction log')

    args = parser.parse_args()
    if args.api_key:
        os.environ['OPENAI_API_KEY'] = args.api_key
    return args
