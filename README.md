# Joint Denoising and Rewiring

This repository contains the code for the paper "Joint Graph Rewiring and Feature Denoising via Spectral Alignment".
The GNN code is based on the ICLR2021 paper Adaptive Universal Generalized PageRank Graph Neural Network [[Paper](https://openreview.net/forum?id=n6jl7fLxrP)] [[Code](https://github.com/jianhao2016/GPRGNN)].


## Baseline Methods
Diffusion Improves Graph Learning (DIGL) [[Paper](https://proceedings.neurips.cc/paper/2019/hash/23c894276a2c5a16470e6a31f4618d73-Abstract.html)] [[Code](https://github.com/gasteigerjo/gdc)]

First-order spectral rewiring (FoSR) [[Paper](https://openreview.net/forum?id=3YjQfCLdrzz)] [[Code](https://github.com/kedar2/FoSR)]

Batch Ollivier-Ricci Flow (BORF) [[Paper](https://proceedings.mlr.press/v202/nguyen23c.html)] [[Code](https://github.com/Fsoft-AIC/Batch-Ollivier-Ricci-Flow/tree/main)]

Approximate Message Passing - Belief Propagation (AMP-BP) [[Paper](https://openreview.net/forum?id=Pe6hldOUkw)] [[Code](https://gitlab.epfl.ch/spoc-idephics/csbm)]

DIGL, FoSR and BORF can be run directly from this repository. For AMP-BP, please refer to the original repository.
# Requirements:
Tested with Python 3.10.14 and PyTorch 2.0.1 (Cuda 11.8).
```
pytorch
pytorch-geometric
numpy scipy matplotlib pyyaml
```
For FoSR and BORF baseline:
```
mamba pandas networkx GraphRicciCurvature
```
Optional (if not used, use flag `--no-wandb_log` when running the code):
```
wandb
```

# Run the Code
In all cases go to folder `src`
### Run GCN+JDR on Cora:

```
python train_model.py --dataset Cora --net GCN --data_split sparse --denoise_default GCN 
```

### Run Rewire Baselines on Cora
DIGL
```
python train_model.py --dataset Cora --net GCN --data_split sparse --rewire_default ppr 
```
FoSR
```
python train_model.py --dataset Cora --net GCN --data_split sparse --rewire_default fosr 
```
BORF
```
python train_model.py --dataset Cora --net GCN --data_split sparse --rewire_default borf 
```
### Reproduce the results of the paper:
```
source run_csbm_exp.sh
```
```
source run_exp_table_1.sh
```
```
source run_exp_table_2.sh
```

# Datasets
## Twitch-gamers
The dataset can be downloaded from [Snap](http://snap.stanford.edu/data/twitch_gamers.html).

## cSBM
To create a new dataset go to folder `src` and run for example:
```
python cSBM_dataset.py --phi 0.6 --name cSBM_phi_0.6 --root ../data/ --num_nodes 5000 --num_features 2000 --avg_degree 5 --epsilon 3.25
```






