import os
from LLMs.LLM import LLM
from configs.inference_config import get_args
import pandas as pd
import csv

if __name__ == '__main__':
    args = get_args()
    pipe = LLM(args)
    descriptions = pd.read_csv(args.description_file).values
    with open(os.path.join(args.result_path, args.project_name, f'sensations_predicted_by_{args.LLM}.csv'), 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['ID', 'description', 'predicted_sensation'])
    for row in descriptions:
        ID = row[0]
        description = row[1]
        prompt = f"""Context: Description of an image is {description}
            Given the description of the image, the sensation that the image evokes is: """
        predicted_sensation = pipe(prompt=prompt)
        print(f'predicted sensation for image {ID} is {predicted_sensation}')
        with open(os.path.join(args.result_path, args.project_name, f'sensations_predicted_by_{args.LLM}.csv'), 'a') as f:
            writer = csv.writer(f)
            writer.writerow([ID, description, predicted_sensation])

