import json
import os


def average_score(metrics_scores):
    values = []
    for image_url in metrics_scores:
        for sensation in metrics_scores[image_url]:
            scores = metrics_scores[image_url][sensation]
            if isinstance(scores, list):
                values.append(scores[-1])
            else:
                values.append(scores)
    print(len(values))
    return sum(values) / len(values)

def compute_average_scores_per_file(metrics_file):
    metric_scores = json.load(open(metrics_file))
    print(f'average scores for {metrics_file} is: {average_score(metric_scores)}')
    print('-' * 100)

def compute_all_average_scores_all_files(directory):
    for filename in os.listdir(directory):
        if filename == '.DS_Store' or '40000' not in filename:
            continue
        metrics_file = os.path.join(directory, filename)
        metrics_file = os.path.join(directory, filename)
        compute_average_scores_per_file(metrics_file)

def compute_average_per_sensation(metrics_scores):
    scores_per_sensation = {}
    average_score_per_sensation = {}
    count = 0
    for image_url in metrics_scores:
        count += 1
        for sensation in metrics_scores[image_url]:
            scores = metrics_scores[image_url][sensation]
            score = scores[-1] if isinstance(scores, list) else scores
            if sensation not in scores_per_sensation:
                scores_per_sensation[sensation] = [score]
            else:
                scores_per_sensation[sensation].append(score)
    for sensation in scores_per_sensation:
        average_score_per_sensation[sensation] = sum(scores_per_sensation[sensation]) / len(scores_per_sensation[sensation])
    return average_score_per_sensation

def compute_average_per_sensations_all_files(directory):
    for filename in os.listdir(directory):
        if filename == '.DS_Store' or '40000' not in filename:
            continue
        print(filename)
        metrics_file = os.path.join(directory, filename)
        metric_scores = json.load(open(metrics_file))
        saving_path = os.path.join('/'.join(directory.split('/')[:-1]), 'average_scores_'+ filename.replace('.json', '_average.json'))
        average_score_per_sensation = compute_average_per_sensation(metric_scores)
        # json.dump(average_score_per_sensation, open(saving_path, 'w'))


def compute_real_images_scores(real_ads, generated_ads):
    image_urls = ['2/36282.jpg', '10/176698.png', '0/135240.jpg', '10/170411.png', '10/170877.png', '1/149281.jpg', '10/176267.png', '0/158280.jpg', '3/17763.jpg', '8/67818.jpg', '7/116247.jpg', '4/92344.jpg']
    real_images_scores = []
    count = 0
    sensation_gen_problem = {}
    list_of_sensations = {}
    for image_url in real_ads:
        if image_url in image_urls:
            continue
        if image_url not in generated_ads:
            continue
        for sensation in real_ads[image_url]:
            if sensation.lower() not in generated_ads[image_url]:
                continue
            if sensation    .lower() in list_of_sensations:
                list_of_sensations[sensation.lower()] += 1
            else:
                list_of_sensations[sensation.lower()] = 1
            scores = real_ads[image_url][sensation]
            if isinstance(scores, list):
                # if scores[-1] > 0.9:

                if scores[-1] < generated_ads[image_url][sensation.lower()][-1]:
                    count += 1
                    print(image_url, sensation, scores[-1], generated_ads[image_url][sensation.lower()][-1])
                else:
                    if sensation.lower() in sensation_gen_problem:
                        sensation_gen_problem[sensation.lower()] += 1
                    else:
                        sensation_gen_problem[sensation.lower()] = 1
                real_images_scores.append(scores[-1])
            else:

                real_images_scores.append(scores)
    print(f'average scores for real ads is: {sum(real_images_scores) / len(real_images_scores)}')
    print(count, len(real_images_scores))
    sorted_dict = dict(sorted(sensation_gen_problem.items(), key=lambda item: item[1], reverse=True))
    print(sorted_dict)
    print(list_of_sensations)



if __name__ == '__main__':
    directory = '/Users/aysanaghazadeh/SensoryAds/Evosense_GT_Sensation'
    compute_all_average_scores_all_files(directory)
    compute_average_per_sensations_all_files(directory)
    # generated_ads = json.load(open('/Users/aysanaghazadeh/experiments/SensoryAds/Evosense_GT_Sensation/IN_InternVL_20250916_122348_AR_ALL_Flux_ALL_description_generation_LLAMA3_instruct_finetunedTrue_21000.json'))
    # real_ads = json.load(open('/Users/aysanaghazadeh/experiments/SensoryAds/Evosense_GT_Sensation/IN_InternVL_20250918_122434_AR_ALL_PixArt_ALL_description_generation_LLAMA3_instruct_finetunedTrue_21000.json'))
    # image_url = '0/56910.jpg'
    # print(generated_ads.keys())
    # print(real_ads[image_url])
    # print(generated_ads[image_url])
    # print('-' * 100)
    # generated_ads = json.load(open(
    #     '/Users/aysanaghazadeh/experiments/SensoryAds/SensoryAds/VQA_score_GT_Sensation/IN_InternVL_20250916_122348_AR_ALL_Flux_ALL_description_generation.json'))
    # real_ads = json.load(open(
    #     '/Users/aysanaghazadeh/experiments/SensoryAds/SensoryAds/VQA_score_GT_Sensation/IN_InternVL_20250918_122434_AR_ALL_PixArt_ALL_description_generation.json'))
    # print(real_ads[image_url])
    # print(generated_ads[image_url])
    # # print(SD3_ads['2/39922.jpg'])
    
    
    