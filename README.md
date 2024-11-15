
# DiffSBDD: Structure-based Drug Design with Equivariant Diffusion Models

This repository contains the implementation of a molecular docking and binding affinity prediction pipeline using equivariant diffusion models and graph neural networks (GNNs). The project leverages EGNN, GAT, and various evaluation metrics to analyze protein-ligand interactions.

# Table of Contents
1. Project Structure
2. Installation
3. Dataset
4. Usage
5. Testing
6. Visualization
7. Configuration
8. Key Components
9. Kaggle/Cloud Environment



## Project Structure

```
├── analysis/
│   ├── SA_Score/
│   │   ├── README.md               
│   │   ├── sascorer.py             
│   │   ├── fpscores.pkl.gz          
│   │   └── __pycache__/             
│   ├── docking.py                   
│   ├── docking_py27.py              
│   ├── metrics.py                  
│   ├── molecule_builder.py          
│   ├── visualization.py            
│   └── __pycache__/                 
├── configs/
│   ├── args-egnn.yml               
│   └── args-gat_hyb.yml            
├── constants.py                    
├── crossdock_data/
│   └──                     
├── dataset.py                       
├── deliverables.txt                 
├── equivariant_diffusion/
│   ├── conditional_model.py         
│   ├── dynamics.py                  
│   ├── egnn_new.py                  
│   ├── en_diffusion.py              
│   └── __pycache__/                 
├── kaggle/
│   ├── diffsbdd-kfold.ipynb         
│   └── diffsbdd-valid-data.ipynb    
├── notebooks/
│   └── viz.ipynb                   
├── kfold_train.py                 
├── lightning_modules.py           
├── optimize.py                    
├── requirements.txt                
├── test.py                         
├── train.py                  
├── utils.py                       
└── README.md                       
```

## Installation

To get started with DiffSBDD, clone the repository and install the necessary dependencies:

```bash
git clone https://github.com/sujaykumarmag/DiffSBDD.git
cd DiffSBDD
pip install -r requirements.txt
```


## Dataset
The prepocessed dataset is available in the Gdrive link (https://drive.google.com/drive/folders/1OySoHpAKGKxhpw9YOQdVf6Lk6XmHdt1T?usp=share_link) and Place the dataset in the `crossdock_data/` directory.


## Usage

### KFold Training with EGNN
To start training the model using k-fold cross-validation (EGNN model):
```bash
python kfold_train.py --config configs/args-egnn.yml
```
### KFold Training with EGNN+GAT
To start training the model using k-fold cross-validation (EGNN + GAT model):
```bash
python kfold_train.py --config configs/args-gat_hyb.yml
```

### Normal Training with EGNN
To start training the model using k-fold cross-validation (EGNN model):
```bash
python train.py --config configs/args-egnn.yml
```
### Normal Training with EGNN+GAT
To start training the model using Usual Training (EGNN + GAT model):
```bash
python train.py --config configs/args-gat_hyb.yml
```




### Testing
To evaluate the model on the test dataset (make sure u provide the model file):
```bash
python metric.py 
```


### Visualization
For visual analysis, open the Jupyter notebook:
```bash
jupyter notebook notebooks/viz.ipynb
```

## Configuration

All model and training configurations are stored in the `configs/` directory. You can modify the `.yml` files to update model parameters, training settings, and data paths.

## Key Components

- **Metrics and Evaluation**: The `metrics.py` script contains various metrics for evaluating model performance.
- **Equivariant Diffusion Models**: Core diffusion models and dynamics implemented in the `equivariant_diffusion/` directory.
- **Equivariant Diffusion Models**: GNN variants with MultiHead Attention, GAT, GIN are also implemented in the `equivariant_diffusion/egnn_new` directory to make sure to have an effective comparison (Subjected to have more GPUs)
- **Training and Testing**: The main training scripts include `kfold_train.py`, `train.py` and `metric.py` 



## Kaggle or Cloud Environment

All files are uploaded in the Siddant kaggle account (but there are problems in installing packages)
Try this code for installing packages else go for the installation provided in the .ipynb file in the `kaggle` directory.

Code file : https://www.kaggle.com/datasets/siddhantgehani/diffsbdd-gat1
Dataset : https://www.kaggle.com/datasets/siddhantgehani/diffsbdd-dataset

```bash
#@title Install dependencies (this will take about 5-10 minutes)
#@title  Install condacolab (the kernel will be restarted, after that you can execute the remaining cells)
!pip install -q condacolab
import condacolab
condacolab.install()

import os

commands = [
    "pip install torch==2.0.1 --extra-index-url https://download.pytorch.org/whl/cu118",
    "pip install pytorch-lightning==1.8.4",
    "pip install wandb==0.13.1",
    "pip install rdkit==2022.3.3",
    "pip install biopython==1.79",
    "pip install imageio==2.21.2",
    "pip install scipy==1.7.3",
    "pip install pyg-lib torch-scatter -f https://data.pyg.org/whl/torch-2.0.1+cu118.html",
    "pip install networkx==2.8.6",
    "pip install py3Dmol==1.8.1",
    "conda install openbabel -c conda-forge",
    "git clone https://github.com/arneschneuing/DiffSBDD.git",
    "mkdir -p /content/DiffSBDD/checkpoints",
    "wget -P /content/DiffSBDD/checkpoints https://zenodo.org/record/8183747/files/moad_fullatom_cond.ckpt",
    "wget -P /content/DiffSBDD/checkpoints https://zenodo.org/record/8183747/files/moad_fullatom_joint.ckpt",
]

errors = {}

if not os.path.isfile("/content/READY"):
  for cmd in commands:
    # os.system(cmd)
    with os.popen(cmd) as f:
      out = f.read()
      status = f.close()

    if status is not None:
      errors[cmd] = out
      print(f"\n\nAn error occurred while running '{cmd}'\n")
      print("Status:\t", status)
      print("Message:\t", out)

if len(errors) == 0:
  os.system("touch /content/READY")
```


## Resuts 

Results are provided in the Gdrive link
https://drive.google.com/drive/folders/1d7UAvvObLkHdkWASaSlDxli6KdUeyuKm?usp=share_link

---

