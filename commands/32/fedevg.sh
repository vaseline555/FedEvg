#!/bin/sh

for s in 1 2 3
do
    for k in 10 100
    do
        echo "Start (K=$k; seed=$s)...!"
        {
            python3 main.py \
            --exp_name "CIFAR10_FedEvg_$s (K=$k; patho=2)" --seed $s --device cuda:0 \
            --dataset CIFAR10 \
            --split_type patho --mincls 2 --test_size 0.2 --bpr 10 --spc 50 \
            --model_name ResNet10 --hidden_size 64 --resize 32 \
            --algorithm fedevg --eval_fraction 1 --eval_type both --eval_every $k --eval_metrics acc1 fid \
            --R 100 --K $k --C $((10 / $k)) --E $((100 / $k)) --B 64 --max_workers $((100 / $k)) \
            --alpha 1. --sigma 0.01 --ld_steps 5 --server_beta 10. --server_beta_last 0.01 --ce_lambda 0.1 \
            --optimizer Adam --lr 0.001 --weight_decay 1e-4 --lr_decay 0.95 --lr_decay_step $k --criterion CrossEntropyLoss &
            sleep 1

            python3 main.py \
            --exp_name "CINIC10_FedEvg_$s (K=$k; patho=2)" --seed $s --device cuda:1 \
            --dataset CINIC10 \
            --split_type patho --mincls 2 --test_size 0.2 --bpr 10 --spc 50 \
            --model_name ResNet10 --hidden_size 64 --resize 32 \
            --algorithm fedevg --eval_fraction 1 --eval_type both --eval_every $k --eval_metrics acc1 fid \
            --R 100 --K $k --C $((10 / $k)) --E $((100 / $k)) --B 64 --max_workers $((100 / $k)) \
            --alpha 1. --sigma 0.01 --ld_steps 5 --server_beta 10. --server_beta_last 0.01 --ce_lambda 0.1 \
            --optimizer Adam --lr 0.001 --weight_decay 1e-4 --lr_decay 0.95 --lr_decay_step $k --criterion CrossEntropyLoss &
            sleep 1

            python3 main.py \
            --exp_name "MNIST_FedEvg_$s (K=$k; patho=2)" --seed $s --device cuda:2 \
            --dataset MNIST \
            --split_type patho --mincls 2 --test_size 0.2 --bpr 10 --spc 50 \
            --model_name ResNet10 --hidden_size 64 --resize 32 \
            --algorithm fedevg --eval_fraction 1 --eval_type both --eval_every $k --eval_metrics acc1 fid \
            --R 100 --K $k --C $((10 / $k)) --E $((100 / $k)) --B 64 --max_workers $((100 / $k)) \
            --alpha 1. --sigma 0.01 --ld_steps 5 --server_beta 10. --server_beta_last 0.01 --ce_lambda 0.1 \
            --optimizer Adam --lr 0.001 --weight_decay 1e-4 --lr_decay 0.95 --lr_decay_step $k --criterion CrossEntropyLoss &
            sleep 1

            python3 main.py \
            --exp_name "CIFAR10_FedEvg_$s (K=$k; unbalanced=0.1)" --seed $s --device cuda:0 \
            --dataset CIFAR10 \
            --split_type unbalanced --kprt 0.1 --test_size 0.2 --bpr 10 --spc 50 \
            --model_name ResNet10 --hidden_size 64 --resize 32 \
            --algorithm fedevg --eval_fraction 1 --eval_type both --eval_every $k --eval_metrics acc1 fid \
            --R 100 --K $k --C $((10 / $k)) --E $((100 / $k)) --B 64 --max_workers $((100 / $k)) \
            --alpha 1. --sigma 0.01 --ld_steps 5 --server_beta 10. --server_beta_last 0.01 --ce_lambda 0.1 \
            --optimizer Adam --lr 0.001 --weight_decay 1e-4 --lr_decay 0.95 --lr_decay_step $k --criterion CrossEntropyLoss &
            sleep 1

            python3 main.py \
            --exp_name "CINIC10_FedEvg_$s (K=$k; unbalanced=0.1)" --seed $s --device cuda:1 \
            --dataset CINIC10 \
            --split_type unbalanced --kprt 0.1 --test_size 0.2 --bpr 10 --spc 50 \
            --model_name ResNet10 --hidden_size 64 --resize 32 \
            --algorithm fedevg --eval_fraction 1 --eval_type both --eval_every $k --eval_metrics acc1 fid \
            --R 100 --K $k --C $((10 / $k)) --E $((100 / $k)) --B 64 --max_workers $((100 / $k)) \
            --alpha 1. --sigma 0.01 --ld_steps 5 --server_beta 10. --server_beta_last 0.01 --ce_lambda 0.1 \
            --optimizer Adam --lr 0.001 --weight_decay 1e-4 --lr_decay 0.95 --lr_decay_step $k --criterion CrossEntropyLoss &
            sleep 1

            python3 main.py \
            --exp_name "MNIST_FedEvg_$s (K=$k; unbalanced=0.1)" --seed $s --device cuda:2 \
            --dataset MNIST \
            --split_type unbalanced --kprt 0.1 --test_size 0.2 --bpr 10 --spc 50 \
            --model_name ResNet10 --hidden_size 64 --resize 32 \
            --algorithm fedevg --eval_fraction 1 --eval_type both --eval_every $k --eval_metrics acc1 fid \
            --R 100 --K $k --C $((10 / $k)) --E $((100 / $k)) --B 64 --max_workers $((100 / $k)) \
            --alpha 1. --sigma 0.01 --ld_steps 5 --server_beta 10. --server_beta_last 0.01 --ce_lambda 0.1 \
            --optimizer Adam --lr 0.001 --weight_decay 1e-4 --lr_decay 0.95 --lr_decay_step $k --criterion CrossEntropyLoss
            sleep 1
        } &&
        echo "...done (K=$k; seed=$s)!"
    done
done