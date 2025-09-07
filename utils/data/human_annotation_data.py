import json
import pandas as pd
images = ['1/76501.jpg', '0/103010.jpg', '0/147220.jpg', '1/67121.jpg', '0/10270.jpg', '10/172774.png', '10/176681.png', '1/67271.jpg', '3/85653.jpg', '0/125600.jpg', '2/12392.jpg', '0/163550.jpg', '10/172643.png', '10/176698.png', '10/174663.png', '0/56910.jpg', '1/34881.jpg', '3/163523.jpg', '0/102540.jpg', '3/9923.jpg', '1/39571.jpg', '3/18053.jpg', '0/10250.jpg', '3/105563.jpg', '1/147761.jpg', '1/166611.jpg', '10/175992.png', '0/119030.jpg', '3/132653.jpg', '3/125433.jpg', '0/34670.jpg', '0/101660.jpg', '0/52690.jpg', '10/174837.png', '2/36282.jpg', '3/133973.jpg', '0/149190.jpg', '3/27813.jpg', '3/76603.jpg', '10/171514.png', '2/39922.jpg', '0/135240.jpg', '10/170147.png', '10/170451.png', '1/166611.jpg', '2/138212.jpg', '0/132580.jpg', '1/44971.jpg', '1/99581.jpg', '1/98281.jpg', '0/96400.jpg', '10/173268.png', '2/78192.jpg', '0/120020.jpg', '10/177623.png', '0/166480.jpg', '0/54100.jpg', '10/172526.png', '3/105343.jpg', '1/85251.jpg', '0/29130.jpg', '2/10032.jpg', '0/10260.jpg', '3/19283.jpg', '0/8800.jpg', '10/176164.png', '10/176605.png', '0/122910.jpg', '10/175919.png', '0/10060.jpg', '10/173329.png', '2/163572.jpg', '0/12050.jpg', '0/32760.jpg', '0/115730.jpg', '0/32750.jpg', '0/89050.jpg', '10/176176.png', '1/98281.jpg', '0/41340.jpg', '10/175189.png', '0/57130.jpg', '10/170411.png', '3/48323.jpg', '10/172335.png', '2/51562.jpg', '10/173969.png', '0/67540.jpg', '0/13050.jpg', '10/173851.png', '10/170274.png', '1/72721.jpg', '0/1530.jpg', '2/53432.jpg', '1/10721.jpg', '10/170486.png', '1/138671.jpg', '0/118910.jpg', '10/170750.png', '1/99531.jpg', '3/10243.jpg', '10/172510.png', '2/77092.jpg', '1/56921.jpg', '3/93973.jpg', '3/67253.jpg', '1/54021.jpg', '10/177817.png', '0/160720.jpg', '0/13110.jpg', '10/176465.png', '10/170877.png', '0/146000.jpg', '1/133691.jpg', '1/88251.jpg', '10/171598.png', '10/173687.png', '10/170652.png', '1/117141.jpg', '0/99650.jpg', '0/80700.jpg', '1/40271.jpg', '2/32432.jpg', '0/111710.jpg', '10/171621.png', '1/149281.jpg', '2/145632.jpg', '10/176267.png', '10/175152.png', '10/171740.png', '10/175336.png', '10/173387.png', '10/170527.png', '10/171339.png', '3/131303.jpg', '1/149241.jpg']


QA = json.load(open('../Data/PittAd/train/QA_Combined_Action_Reason_train.json'))
test_set_QA = json.load(open('../Data/PittAd/test/QA_Combined_Action_Reason_test.json'))
generated_data_COM = pd.read_csv('/Users/aysanaghazadeh/experiments/results/AR_Flux_20250820_231732.csv')
generated_data_PSA = pd.read_csv('/Users/aysanaghazadeh/experiments/results/LLM_input_physical_sensation_LLAMA3_instruct_FTFalse_PSA.csv')
print(len(images))
print(len(generated_data_COM.image_url.values))
print(len(generated_data_PSA.ID.values))
new_QA = {}
for image in images:
    if image in test_set_QA:
        new_QA[image] = test_set_QA[image]
    else:
        new_QA[image] = QA[image]
for image in generated_data_COM.image_url.values:
    new_QA[image] = QA[image]
for image in generated_data_PSA.ID.values:
    new_QA[image] = QA[image]
print(len(new_QA))
with open('../Data/PittAd/train/QA_Combined_Action_Reason_human_annotation_set.json', 'w') as f:
    json.dump(new_QA, f)
    

