import os, shutil
import json
import csv

os.makedirs('../Data/RLHF_DATA_sensation_aim', exist_ok=True)
os.makedirs('../Data/RLHF_DATA_sensation_aim/train', exist_ok=True)
os.makedirs('../Data/RLHF_DATA_sensation_aim/train/Flux', exist_ok=True)
os.makedirs('../Data/RLHF_DATA_sensation_aim/train/AuraFlow', exist_ok=True)
flux_images_aim = json.load(open('../experiments/results/SensoryAds/IN_InternVL_20250916_122348_AR_ALL_Flux_ALL_description_generationLLAMA3_instruct_text_image_alignment_isFineTunedTrue_3000_weighted.json'))
auraflow_images_aim = json.load(open('../experiments/results/SensoryAds/IN_InternVL_20250916_220717_AR_ALL_AuraFlow_ALL_description_generationLLAMA3_instruct_text_image_alignment_isFineTunedTrue_3000_weighted.json'))
flux_images_sensation = json.load(open('../experiments/results/SensoryAds/Evosense_GT_Sensation/IN_InternVL_20250916_122348_AR_ALL_Flux_ALL_description_generation_LLAMA3_instruct_finetunedTrue_21000.json'))
auraflow_images_sensation = json.load(open('../experiments/results/SensoryAds/Evosense_GT_Sensation/IN_InternVL_20250916_220717_AR_ALL_AuraFlow_ALL_description_generation_LLAMA3_instruct_finetunedTrue_21000.json'))
auraflow_image_path = '../experiments/generated_images/SensoryAds/20250916_220717/AR_ALL_AuraFlow'
flux_image_path = '../experiments/generated_images/SensoryAds/20250916_122348/AR_ALL_Flux'
saving_file = '../Data/RLHF_DATA_sensation_aim/train_processed_v3.csv'
saving_path = '../Data/RLHF_DATA_sensation_aim'
QA = json.load(open('../Data/PittAd/train/QA_Combined_Action_Reason_train.json'))
fieldnames = ['file_name', 'caption', 'split', 'prompt']
data = []
for image_url in flux_images_aim:
    sensation = image_url.split('/')[0]
    image_url = '/'.join(image_url.split('/')[-2:])
    print(image_url)
    print(sensation)
    if (f'{sensation}/ {image_url}' not in auraflow_images_aim
            or f'{sensation}/ {image_url}' not in auraflow_images_sensation
            or f'{sensation}/ {image_url}' not in flux_images_sensation
            or image_url not in QA):
        continue
    AR = QA[image_url][0]
    print(AR)
    AR = '\n -'.join(AR)
    prompt = f'Generate an advertisement image that evokes the {sensation} sensation and conveys the following messages: \n {AR}'
    file_name_auraflow = f'train/AuraFlow/{image_url}'
    file_name_flux = f'train/Flux/{image_url}'
    shutil.copyfile(f'{auraflow_image_path}/{sensation}/{image_url}', f'{saving_path}/{file_name_auraflow}')
    shutil.copyfile(f'{flux_image_path}/{sensation}/{image_url}', f'{saving_path}/{file_name_flux}')
    aim_score_auraflow = auraflow_images_aim[f'{sensation}/{image_url}'][1]
    aim_score_flux = flux_images_aim[f'{sensation}/{image_url}'][1]
    sensation_score_auraflow = auraflow_images_sensation[f'{sensation}/{image_url}'][-1]
    sensation_score_flux = flux_images_sensation[f'{sensation}/{image_url}'][-1]
    auraflow_score = (aim_score_auraflow * 3 + sensation_score_auraflow) / 4
    flux_score = (aim_score_flux * 3 + sensation_score_flux) / 4
    flux_split = 'exclusive_lose' if flux_score < auraflow_score else 'exclusive_win'
    auraflow_split = 'exclusive_lose' if auraflow_score < flux_score else 'exclusive_win'
    flux_data = {'file_name': file_name_flux, 'caption': prompt, 'split': flux_split, 'prompt': prompt}
    auraflow_data = {'file_name': file_name_auraflow, 'caption': prompt, 'split': auraflow_split, 'prompt': prompt}
    data.append(flux_data)
    data.append(auraflow_data)
with open(saving_file, 'w', newline='') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader() # Write the header row
    writer.writerows(data)





