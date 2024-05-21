#! /bin/sh
#
# run_all_model_dense_split_demo.sh

for net in GCN GPRGNN
do
  for dataset in cSBM_phi_-1.0 cSBM_phi_-0.875 cSBM_phi_-0.75 cSBM_phi_-0.625 cSBM_phi_-0.5 cSBM_phi_-0.375 cSBM_phi_-0.25 cSBM_phi_-0.125 cSBM_phi_0.0 cSBM_phi_0.125 cSBM_phi_0.25 cSBM_phi_0.375 cSBM_phi_0.5 cSBM_phi_0.625 cSBM_phi_0.75 cSBM_phi_0.875 cSBM_phi_1.0

  do
      python train_model.py --RPMAX 2 --dataset $dataset --data_split sparse --net $net --no-wandb_log
      python train_model.py --RPMAX 2 --dataset $dataset --data_split sparse --net $net --no-wandb_log --denoise_default $net
      python train_model.py --RPMAX 2 --dataset $dataset --data_split sparse --net $net --no-wandb_log --rewire_default ppr
  done
done