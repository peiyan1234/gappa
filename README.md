# GAPPA: Enhancing Prognosis Prediction in Primary Aldosteronism Post-Adrenalectomy Using Graph-Based Modeling

### About
------------
This repository is the official PyTorch implementation of GAPPA. 
Please cite this paper if you applied this work.

GAPPA: Enhancing prognosis prediction in primary aldosteronism post-adrenalectomy using graph-based modeling
Artificial Intelligence in Medicine
2025-01 | Journal article
DOI: 10.1016/j.artmed.2024.103028
Contributors: Pei-Yan Li; Yu-Wen Huang; Vin-Cent Wu; Jeff S. Chueh; Chi-Shin Tseng; Chung-Ming Chen

### Usages:
------------

To train:
```bash
python train.py pa
```

To test:
```bash
python infer.py pa
```

### Pre-requisites:
- Ubuntu 22.04.2 LTS (GNU/Linux 6.5.0-44-generic x86_64)
- NVIDIA GPU (Tested on Nvidia GeForce RTX 3090 ) with CUDA Version: 12.0
- Python (3.9.13)
- Anaconda version 22.9.0

### Installation:
------------
```bash
conda env create -f environment.yml
conda activate gappa_env

more information:
If you have problems with the installation of PyTorch or PyG, please refer to the official pages:
Install [PyTorch](https://pytorch.org/)
Install [PyTorch_Geometric](https://rusty1s.github.io/pytorch_geometric/build/html/notes/installation.html)
