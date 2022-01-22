# Implementation of WWW 2022 paper: Conditional Generation Net for Medication Recommendation

### Folder Specification
- mimic_env.yaml
- src/
    - COGNet.py: train/test COGNet
    - recommend.py: some test function used for COGNet
    - COGNet_modelt.py: full model of COGNet
    - COGNet_ablation.py: ablation models of COGNet
    - train/test baselines:
        - MICRON.py
        - Other code of train/test baselines can be find [here](https://github.com/ycq091044/SafeDrug).
    - models.py: baseline models
    - util.py
    - layer.py
- data/ **(For a fair comparision, we use the same data and pre-processing scripts used in [Safedrug](https://github.com/ycq091044/SafeDrug))**
    - mapping files that collected from external sources
        - drug-atc.csv: drug to atc code mapping file
        - drug-DDI.csv: this a large file, could be downloaded from https://drive.google.com/file/d/1mnPc0O0ztz0fkv3HF-dpmBb8PLWsEoDz/view?usp=sharing
        - ndc2atc_level4.csv: NDC code to ATC-4 code mapping file
        - ndc2rxnorm_mapping.txt: NDC to xnorm mapping file
        - idx2drug.pkl: drug ID to drug SMILES string dict
    - other files that generated from mapping files and MIMIC dataset (we attach these files here, user could use our provided scripts to generate)
        - data_final.pkl: intermediate result
        - ddi_A_final.pkl: ddi matrix
        - ddi_matrix_H.pkl: H mask structure (This file is created by ddi_mask_H.py), used in Safedrug baseline
        - idx2ndc.pkl: idx2ndc mapping file
        - ndc2drug.pkl: ndc2drug mapping file
        - Under MIMIC Dataset policy, we are not allowed to distribute the datasets. Practioners could go to https://physionet.org/content/mimiciii/1.4/ and requrest the access to MIMIC-III dataset and then run our processing script to get the complete preprocessed dataset file.
        - voc_final.pkl: diag/prod/med dictionary
    - dataset processing scripts
        - processing.py: is used to process the MIMIC original dataset.




### Step 1: Data Processing

- Go to https://physionet.org/content/mimiciii/1.4/ to download the MIMIC-III dataset (You may need to get the certificate)

- go into the folder and unzip three main files (PROCEDURES_ICD.csv.gz, PRESCRIPTIONS.csv.gz, DIAGNOSES_ICD.csv.gz)

- change the path in processing.py and processing the data to get a complete records_final.pkl

  ```python
  vim processing.py
  
  # line 310-312
  # med_file = '/data/mimic-iii/PRESCRIPTIONS.csv'
  # diag_file = '/data/mimic-iii/DIAGNOSES_ICD.csv'
  # procedure_file = '/data/mimic-iii/PROCEDURES_ICD.csv'
  
  python processing.py
  ```

- run ddi_mask_H.py to get the ddi_mask_H.pkl

  ```python
  python ddi_mask_H.py
  ```



### Step 2: Package Dependency

- First, install the [conda](https://www.anaconda.com/)

- Then, create the conda environment through yaml file
```python
conda env create -f mimic_env.yaml
```


### Step 3: run the code

```python
python COGNet.py
```

here is the argument:

    usage: COGNet.py [-h] [--Test] [--model_name MODEL_NAME]
                   [--resume_path RESUME_PATH] [--lr LR]
                   [--target_ddi TARGET_DDI] [--kp KP] [--dim DIM]
    
    optional arguments:
      -h, --help            show this help message and exit
      --Test                test mode
      --model_name MODEL_NAME
                            model name
      --resume_path RESUME_PATH
                            resume path
      --lr LR               learning rate
      --batch_size          batch size 
      --emb_dim             dimension size of embedding
      --max_len             max number of recommended medications
      --beam_size           number of ways in beam search

If you cannot run the code on GPU, just change line 61, "cuda" to "cpu".

### Citation
```bibtex
@inproceedings{wu2022cognet,
    title = {Conditional Generation Net for Medication Recommendation},
    author = {Rui Wu, Zhaopeng Qiu, Jiacheng Jiang, Guilin Qi, and Xian Wu.},
    booktitle = {{WWW} '22: The Web Conference 2022, Virtual Event, Lyon, France, April 25-29, 2022},
    year = {2022}
}
```

Please feel free to contact me <RhysWu@outlook.com> for any question.

Partial credit to previous reprostories:
- https://github.com/sjy1203/GAMENet
- https://github.com/ycq091044/SafeDrug
- https://github.com/ycq091044/MICRON

Thank [Chaoqi Yang](https://github.com/ycq091044) and [Junyuan Shang](https://github.com/sjy1203) for releasing their codes!

Thank my mentor, [Zhaopeng Qiu](https://github.com/zpqiu), for helping me complete the code.
