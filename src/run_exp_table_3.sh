#! /bin/sh
#
# run_all_model_dense_split_demo.sh

for net in GCN
do
  for dataset in Questions Penn94 Twitch-gamers
  do
      python train_model.py --dataset $dataset --data_split sparse --net $net --no-wandb_log
      python train_model.py --dataset $dataset --data_split sparse --net $net --no-wandb_log --rewire_default ppr
      python train_model.py --dataset $dataset --data_split sparse --net $net --no-wandb_log --rewire_default fosr
      python train_model.py --dataset $dataset --data_split sparse --net $net --no-wandb_log --denoise_default $net
  done
done