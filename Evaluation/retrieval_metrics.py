import json

def get_precision_per_image(sensation_scores, model_sensations):
    correct_count = 0
    for sensation in model_sensations:
        sensation = sensation.replace('  ', ' ')
        sensation = sensation.replace('natural greenary smell', 'natural greenery smell')
        if sensation.lower() == 'pouring sound':
            sensation = 'liquid pouring sound'
        if sensation.lower() == 'splash sound':
            sensation = 'liquid splash sound'
        if sensation.lower() == 'traffic jam sound':
            sensation = 'Traffic Jam or Human Crowd Sound'.lower()
        if sensation.lower() == 'heavinesstension':
            sensation = 'heaviness'
        if sensation_scores[sensation] > 0:
            correct_count += 1
    if len(model_sensations) == 0:
        return 1
    return float(correct_count) / len(model_sensations)

def get_recall_per_image(sensation_scores, model_sensations):
    correct_count = 0
    target_sensation_count = 0
    for sensation in sensation_scores:
        sensation = sensation.replace('  ', ' ')
        if sensation_scores[sensation] > 0:
            target_sensation_count += 1
            if sensation in model_sensations:
                correct_count += 1
    if target_sensation_count == 0:
        return 1
    return float(correct_count) / target_sensation_count

def evaluate_retrieval(sensation_annotations, model_results):
    precision, recall, f1_score, count = 0, 0, 0, 0
    for image_url in model_results:
        if image_url not in sensation_annotations:
            continue
        count += 1
        model_predicted_sensations = set()
        for prediction_set in model_results[image_url]:
            predictions = set(prediction_set.lower().split(','))
            model_predicted_sensations.update(predictions)
        image_precision = get_precision_per_image(sensation_annotations[image_url]['sensation_scores'], model_predicted_sensations)
        image_recall = get_recall_per_image(sensation_annotations[image_url]['sensation_scores'], model_predicted_sensations)
        if image_precision + image_recall == 0:
            image_f1_score = 0
        else:
            image_f1_score = (image_precision * image_recall) / (image_precision + image_recall)
        # print('image_url', image_url)
        # print('image_precision', image_precision)
        # print('image_recall', image_recall)
        # print('image_f1_score', image_f1_score)
        precision += image_precision
        recall += image_recall
        f1_score += image_f1_score
    precision /= count
    recall /= count
    f1_score /= count
    return precision, recall, f1_score

if __name__ == '__main__':
    results = [
        'sensation_extraction_PittAd_ALL_QWenVL_MLLM_Sensation_Retrieval.json',
        'sensation_extraction_PittAd_ALL_QWenLM_LLM_Sensation_Retrieval_QWenVL.json',
        'sensation_extraction_PittAd_ALL_QWenLM_LLM_Sensation_Retrieval_InternVL.json',
        'sensation_extraction_PittAd_ALL_QWenLM_LLM_Sensation_Retrieval_Gemma.json',
        'sensation_extraction_PittAd_ALL_LLAVA16_MLLM_Sensation_Retrieval.json',
        'sensation_extraction_PittAd_ALL_LLAMA3_instruct_LLM_Sensation_Retrieval_QWenVL.json',
        'sensation_extraction_PittAd_ALL_LLAMA3_instruct_LLM_Sensation_Retrieval_InternVL.json',
        'sensation_extraction_PittAd_ALL_LLAMA3_instruct_LLM_Sensation_Retrieval_Gemma.json',
        'sensation_extraction_PittAd_ALL_InternVL_MLLM_Sensation_Retrieval.json',
        'sensation_extraction_PittAd_ALL_Gemma_MLLM_Sensation_Retrieval.json',
        'sensation_extraction_PittAd_ALL_Gemma_LLM_Sensation_Retrieval_Gemma.json',
        'sensation_extraction_PittAd_ALL_Gemma_LLM_Sensation_Retrieval_InternVL.json',
        'sensation_extraction_PittAd_ALL_Gemma_LLM_Sensation_Retrieval_QWenVL.json'
    ]
    sensation_annotations = json.load(open('/Users/aysanaghazadeh/Downloads/sensation_annotations_parsed.json'))
    for filename in results:
        model_results = json.load(open(f'/Users/aysanaghazadeh/experiments/results/SensoryAds/SensoryAds/{filename}'))
        precision, recall, f1_score = evaluate_retrieval(sensation_annotations, model_results)
        print(filename)
        print('precision', precision)
        print('recall', recall)
        print('f1_score', f1_score)
        print('-'*100)