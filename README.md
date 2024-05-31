# Prospects for AI-Enhanced ECG as a Unified Screening Tool for Cardiac and Non-Cardiac Conditions -- An Explorative Study in Emergency Care


This repository hosts the code of the paper [Prospects for AI-Enhanced ECG as a Unified Screening Tool for Cardiac and Non-Cardiac Conditions -- An Explorative Study in Emergency Care](https://academic.oup.com/ehjdh/advance-article/doi/10.1093/ehjdh/ztae039/7670685). <ins>Accepted by European Heart Journal Digital Health.</ins>
In this study we introduced a unified deep learning model for ECG analysis, predicting a wide range of cardiac and non-cardiac discharge diagnoses based on the ICD10 classification system with impressive AUROC scores. Our approach excels in handling diverse diagnostic scenarios, suggesting its use as a screening tool in emergency departments, integrated into clinical decision support systems. <ins>We therefore propose the MIMIC-IV-ECG-ICD-ED dataset derived from the MIMIC-IV and MIMIC-IV-ECG databases primarily for benchmark purposes.</ins>

[![arXiv](https://img.shields.io/badge/arXiv-1234.56789-b31b1b.svg)](https://arxiv.org/abs/2312.11050)


## Benchmarking scenarios. ED subset [T(ED2ALL)-E(ED2ALL)] is the primarly scenario discussed throught the main text in the manuscript:
![alt text](https://github.com/AI4HealthUOL/ECG-MIMIC/blob/main/reports/mimic_benchmark.png?style=centerme)


## ED subset and MIMIC-IV-ECG-ICD-ED statistics:
![alt text](https://github.com/AI4HealthUOL/ECG-MIMIC/blob/main/reports/description.png?style=centerme)


## MIMIC-IV-ECG-ICD-ED statements-distributions:
(A) represents the distribution of statements according to chapters (all percentages as relative fractions compared to the dataset size), whereas (B) represents the distribution of cardiac conditions within chapter IX.
![alt text](https://github.com/AI4HealthUOL/ECG-MIMIC/blob/main/reports/dataset_all-1.png?style=centerme)


## Main ED use case investigated in manuscript statements-distributions:
(A) represents the distribution of statements according to chapters (all percentages as relative fractions compared to the dataset size), whereas (B) represents the distribution of cardiac conditions within chapter IX. However, these are the distributions of a specific ED use case (subset dataset) investigated in the manuscript.
![alt text](https://github.com/AI4HealthUOL/ECG-MIMIC/blob/main/reports/dataset_ed-1.png?style=centerme)



## Datasets and experiments

### 1.0 Datasets download

Download the [MIMIC-IV-ECG](https://physionet.org/content/mimic-iv-ecg/1.0/) dataset and the [MIMIC-IV](https://physionet.org/content/mimiciv/2.2/) dataset (with credentialed access).


### 1.1 Datasets preprocessing

Go under src/ and run the following command where your should replace the corresponding data paths

```
python full_preprocessing.py --mimic-path <path to mimic-iv directory ended in 'mimiciv/2.2/'> --zip-path <path to ecgs zip file> --target-path <desired output for preprocessed data default='./'>
```

### 1.2 Models training

These are 2 of the total benchmarks commands, T(ED2ALL)-E(ED2ALL) the main scenario thtought the main text, and T(ALL2ALL)-E(ALL2ALL) the complete dataset. These command should also export your test set predictions into a corresponding path directory (already specified in a command argument), and also save resulting AUROCs in an also specified log file.

Optinally, see src/demo.ipynb for an example of how to acess each of specific bencharmking scenario.



T(ED2ALL)-E(ED2ALL)

```
python main_ecg.py --data <your data path> --input-size 250 --finetune-dataset mimic_ed_all_edfirst_all_2000_5A --architecture s4 --precision 32 --s4-n 8 --s4-h 512 --batch-size 32 --epochs 20 --export-predictions-path T(ED2ALL)-E(ED2ALL)/ > T(ED2ALL)-E(ED2ALL).log
```

T(ALL2ALL)-E(ALL2ALL)

```
python main_ecg.py --data <your data path> --input-size 250 --finetune-dataset mimic_all_all_allfirst_all_2000_5A --architecture s4 --precision 32 --s4-n 8 --s4-h 512 --batch-size 32 --epochs 20 --export-predictions-path T(ALL2ALL)-E(ALL2ALL)/ > T(ALL2ALL)-E(ALL2ALL).log
```


### 2.0 Physionet direct download (to be done soon)



## Results

You can find all the experimental results for each of the labels and scenarios under reports/Results_MIMIC_IV_ECG_ICD.csv




## Reference

```bibtex
@article{10.1093/ehjdh/ztae039,
    author = {Strodthoff, Nils and Lopez Alcaraz, Juan Miguel and Haverkamp, Wilhelm},
    title = "{Prospects for AI-Enhanced ECG as a Unified Screening Tool for Cardiac and Non-Cardiac Conditions – An Explorative Study in Emergency Care}",
    journal = {European Heart Journal - Digital Health},
    pages = {ztae039},
    year = {2024},
    month = {05},
    abstract = "{Current deep learning algorithms designed for automatic ECG analysis have exhibited notable accuracy. However, akin to traditional electrocardiography, they tend to be narrowly focused and typically address a singular diagnostic condition. In this exploratory study, we specifically investigate the capability of a single model to predict a diverse range of both cardiac and non-cardiac discharge diagnoses based on a sole ECG collected in the emergency department. We find that 253, 81 cardiac and 172 non-cardiac, ICD codes can be reliably predicted in the sense of exceeding an AUROC score of 0.8 in a statistically significant manner. This underscores the model’s proficiency in handling a wide array of cardiac and non-cardiac diagnostic scenarios which demonstrates potential as a screening tool for diverse medical encounters.}",
    issn = {2634-3916},
    doi = {10.1093/ehjdh/ztae039},
    url = {https://doi.org/10.1093/ehjdh/ztae039},
    eprint = {https://academic.oup.com/ehjdh/advance-article-pdf/doi/10.1093/ehjdh/ztae039/57553846/ztae039.pdf},
}
```

