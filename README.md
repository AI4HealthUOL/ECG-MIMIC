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

1. Datasets download



2. Datasets preprocessing



3. Models training


T(ED2ALL)-E(ED2ALL)

```
python main_ecg.py --data <your data path> --input-size 250 --finetune-dataset mimic_ed_all_edfirst_all_2000_5A --architecture s4 --precision 32 --s4-n 8 --s4-h 512 --batch-size 32 --epochs 20 --export-predictions-path T(ED2ALL)-E(ED2ALL)/ > T(ED2ALL)-E(ED2ALL).log
```

T(ALL2ALL)-E(ALL2ALL)

```
python main_ecg.py --data <your data path> --input-size 250 --finetune-dataset mimic_all_all_allfirst_all_2000_5A --architecture s4 --precision 32 --s4-n 8 --s4-h 512 --batch-size 32 --epochs 20 --export-predictions-path T(ALL2ALL)-E(ALL2ALL)/ > T(ALL2ALL)-E(ALL2ALL).log
```

## Results
You can find all the experimental results for each of the labels and scenarios under reports/Results_MIMIC_IV_ECG_ICD.csv





```bibtex
@article{
lopez alcaraz2022diffusionbased,
title={Diffusion-based Time Series Imputation and Forecasting with Structured State Space Models},
author={Juan Lopez Alcaraz and Nils Strodthoff},
journal={Transactions on Machine Learning Research},
issn={2835-8856},
year={2022},
url={https://openreview.net/forum?id=hHiIbk7ApW},
}

```

