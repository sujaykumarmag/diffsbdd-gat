
# DiffSBDD: Structure-based Drug Design with Equivariant Diffusion Models

DiffSBDD is a cutting-edge project focused on leveraging equivariant diffusion models and graph neural networks (GNNs) for predicting molecular docking and binding affinities in structure-based drug design. This repository provides the tools and scripts necessary for training, evaluating, and visualizing models for protein-ligand interaction prediction.

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

## Usage

### Training
To start training the model using k-fold cross-validation:
```bash
python kfold_train.py --config configs/args-egnn.yml
```

### Testing
To evaluate the model on the test dataset:
```bash
python test.py --config configs/args-gat_hyb.yml
```

### Hyperparameter Optimization
You can optimize hyperparameters using:
```bash
python optimize.py
```

### Visualization
For visual analysis, open the Jupyter notebook:
```bash
jupyter notebook notebooks/viz.ipynb
```

## Configuration

All model and training configurations are stored in the `configs/` directory. You can modify the `.yml` files to update model parameters, training settings, and data paths.

## Key Components

- **SA_Score Module**: Calculates the Synthetic Accessibility (SA) score of generated molecules using `sascorer.py`.
- **Docking Analysis**: The `docking.py` script analyzes molecular docking results, with support for Python 2.7 using `docking_py27.py`.
- **Metrics and Evaluation**: The `metrics.py` script contains various metrics for evaluating model performance.
- **Equivariant Diffusion Models**: Core diffusion models and dynamics implemented in the `equivariant_diffusion/` directory.
- **Training and Testing**: The main training scripts include `kfold_train.py` and `test.py`.

## Deliverables

See `deliverables.txt` for a comprehensive list of project deliverables and milestones.

## Dependencies

All dependencies are listed in `requirements.txt`. Install them using:
```bash
pip install -r requirements.txt
```

## License

This project is licensed under the MIT License. See the `LICENSE` file for more information.

## Contact

For any questions or feedback, please contact the project maintainer.

---

