import os
import shutil

models_paths = {
    'SD3': '../experiments/generated_images/SensoryAds/20250916_130149/AR_ALL_SD3',
    'FLUX': '../experiments/generated_images/SensoryAds/20250916_122348/AR_ALL_Flux',
    'QWenImage': '../experiments/generated_images/SensoryAds/20250917_185403/AR_ALL_QWenImage',
    'PixArt': '../experiments/generated_images/SensoryAds/20250918_122434/AR_ALL_PixArt',
    'AuraFlow': '../experiments/generated_images/SensoryAds/20250916_220717/AR_ALL_AuraFlow'
}
image_list = ['freezing cold/1/166611.jpg',
              'brilliance and glow/0/101660.jpg',
              'sweet taste/2/36282.jpg',
              'natural greenery smell/0/103010.jpg',
              'dryness/10/172643.png',
              'culinary smell/0/56910.jpg',
              'roughness/3/133973.jpg',
              'liquid splash sound/10/175992.png',
              'soaking wetness/3/105563.jpg',
              'high speed and acceleration/0/119030.jpg',
              'spicy taste/1/67121.jpg',
              'cool and refreshing/1/34881.jpg',
              'darkness/4/154234.jpg',
              'aching pain/4/41434.jpg'
              'pungent smell/9/116199.jpg'
              ]
public_path = '../../public/html/generated_images'
for model_id in models_paths:
    model_path = models_paths[model_id]
    for image_url in image_list:
        image_path = os.path.join(model_path, image_url)
        shutil.copy(image_path, os.path.join(public_path, f'{model_id}_{image_url}'))
for filename in os.listdir(public_path):
    print(filename, 'copied')