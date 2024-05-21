#! /bin/sh
#
# run_all_model_dense_split_demo.sh

for net in GCN GPRGNN
do
  for dataset in Cora Citeseer Pubmed Computers Photo
  do
      python train_model.py --dataset $dataset --data_split sparse --net $net --no-wandb_log
      python train_model.py --dataset $dataset --data_split sparse --net $net --no-wandb_log --denoise_default $net
      python train_model.py --dataset $dataset --data_split sparse --net $net --no-wandb_log --rewire_default borf
      python train_model.py --dataset $dataset --data_split sparse --net $net --no-wandb_log --rewire_default ppr
  done
done