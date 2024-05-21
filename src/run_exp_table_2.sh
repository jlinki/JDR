#! /bin/sh
#
# run_all_model_dense_split_demo.sh

for net in GCN GPRGNN
do
  for dataset in Chameleon Squirrel Film Texas Cornell
  do
      python train_model.py --dataset $dataset --data_split dense --net $net --no-wandb_log
      python train_model.py  --dataset $dataset --data_split dense --net $net --no-wandb_log --denoise_default $net
      if [ "$dataset" != "Squirrel" ]; then
        python train_model.py --dataset $dataset --data_split dense --net $net --no-wandb_log --rewire_default borf
      fi
      python train_model.py --dataset $dataset --data_split dense --net $net --no-wandb_log --rewire_default ppr
  done
done