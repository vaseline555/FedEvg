#!/bin/sh

python3 main.py \
    --exp_name "MNIST_gradient_attack_scenario" --seed 0 --device cuda:0 \
    --dataset MNIST \
    --split_type patho --mincls 2 --test_size 0.2 --server_sync 10 --spc 10 --bpr 10 --penult_spectral_norm \
    --model_name ResNet10 --hidden_size 64 --resize 32 \
    --algorithm fedevg --eval_fraction 1 --eval_type both --eval_every 10 --eval_metrics acc1 fid \
    --R 100 --K 10 --C 1 --E 1 --B 64 \
    --alpha 1. --sigma 0.01 --ld_steps 5 --cd_init pcd --server_beta 10. --server_beta_last 1. --ce_lambda 0.1 \
    --optimizer Adam --lr 0.001 --weight_decay 1e-4 --lr_decay 0.95 --lr_decay_step 10 --criterion CrossEntropyLoss