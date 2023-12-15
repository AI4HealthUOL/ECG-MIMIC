# Cardiac and extracardiac discharge diagnosis prediction from emergency department ECGs using deep learning


This repository hosts the code of the paper [Cardiac and extracardiac discharge diagnosis prediction from emergency department ECGs using deep learning](https://openreview.net/forum?id=hHiIbk7ApW&referrer=%5BTMLR%5D(%2Fgroup%3Fid%3DTMLR)). 
In this study we introduced a unified deep learning model for ECG analysis, predicting a wide range of cardiac and non-cardiac discharge diagnoses based on the ICD10 classification system with impressive AUROC scores. Our approach excels in handling diverse diagnostic scenarios, suggesting its use as a screening tool in emergency departments, integrated into clinical decision support systems. We therefore propose the MIMIC-IV-ECG-ICD dataset derived from the MIMIC-IV and MIMIC-IV-ECG databases. 

## MIMIC-IV-ECG-ICD experimental workflow:
![alt text](https://github.com/AI4HealthUOL/ECG-MIMIC/blob/main/reports/pipeline_mimic-1.png?style=centerme)


## MIMIC-IV-ECG-ICD statements-distributions:
(A) represents the distribution of statements according to chapters (all percentages as relative fractions compared to the dataset size), whereas (B) represents the distribution of cardiac conditions within chapter IX.
![alt text](https://github.com/AI4HealthUOL/ECG-MIMIC/blob/main/reports/dataset_all-1.png?style=centerme)


## Main ED use case investigated in manuscript statements-distributions:
(A) represents the distribution of statements according to chapters (all percentages as relative fractions compared to the dataset size), whereas (B) represents the distribution of cardiac conditions within chapter IX. However, these are the distributions of a specific ED use case (subset dataset) investigated in the manuscript.
![alt text](https://github.com/AI4HealthUOL/ECG-MIMIC/blob/main/reports/dataset_ed-1.png?style=centerme)


## ED subset and MIMIC-IV-ECG-ICD statistics:
![alt text](https://github.com/AI4HealthUOL/ECG-MIMIC/blob/main/reports/description.png?style=centerme)


## Deep-learning investigated architectures:
(A) XResNet1d50 (B) S4
![alt text](https://github.com/AI4HealthUOL/ECG-MIMIC/blob/main/reports/architectures-1.png?style=centerme)


## Datasets and experiments



## Results
You can find all the experimental results for each of the labels and scenarios under ::reports/Results_MIMIC_IV_ECG_ICD.csv::
