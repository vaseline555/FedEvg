#!/bin/sh

for s in 1 2 3
do
    for k in 10 100
    do
        echo "Start (K=$k; seed=$s)...!"
        {
            python3 main.py \
            --exp_name "CIFAR10_FedCGAN_$s (K=$k; diri=0.01)" --seed $s --device cuda:0 \
            --dataset CIFAR10 \
            --split_type diri --cncntrtn 0.01 --test_size 0.2 --spc 50 \
            --model_name ACGAN --hidden_size 64 --resize 32 --init_type normal --init_gain 0.02 \
            --algorithm fedcgan --eval_fraction 1 --eval_type both --eval_every 10 --eval_metrics acc1 fid \
            --R $((10 * $k)) --K $k --C $((10 / $k)) --E 1 --B 64 --max_workers $((100 / $k)) \
            --optimizer Adam --lr 0.001 --weight_decay 1e-4 --lr_decay 0.95 --lr_decay_step $k --criterion CrossEntropyLoss &
            sleep 1

            python3 main.py \
            --exp_name "CINIC10_FedCGAN_$s (K=$k; diri=0.01)" --seed $s --device cuda:1 \
            --dataset CINIC10 \
            --split_type diri --cncntrtn 0.01 --test_size 0.2 --spc 50 \
            --model_name ACGAN --hidden_size 64 --resize 32 --init_type normal --init_gain 0.02 \
            --algorithm fedcgan --eval_fraction 1 --eval_type both --eval_every 10 --eval_metrics acc1 fid \
            --R $((10 * $k)) --K $k --C $((10 / $k)) --E 1 --B 64 --max_workers $((100 / $k)) \
            --optimizer Adam --lr 0.001 --weight_decay 1e-4 --lr_decay 0.95 --lr_decay_step $k --criterion CrossEntropyLoss &
            sleep 1

            python3 main.py \
            --exp_name "MNIST_FedCGAN_$s (K=$k; diri=0.01)" --seed $s --device cuda:2 \
            --dataset MNIST \
            --split_type diri --cncntrtn 0.01 --test_size 0.2 --spc 50 \
            --model_name ACGAN --hidden_size 64 --resize 32 --init_type normal --init_gain 0.02 \
            --algorithm fedcgan --eval_fraction 1 --eval_type both --eval_every 10 --eval_metrics acc1 fid \
            --R $((10 * $k)) --K $k --C $((10 / $k)) --E 1 --B 64 --max_workers $((100 / $k)) \
            --optimizer Adam --lr 0.001 --weight_decay 1e-4 --lr_decay 0.95 --lr_decay_step $k --criterion CrossEntropyLoss &
            sleep 1

            python3 main.py \
            --exp_name "CIFAR10_FedCGAN_$s (K=$k; diri=1.00)" --seed $s --device cuda:0 \
            --dataset CIFAR10 \
            --split_type diri --cncntrtn 1.00 --test_size 0.2 --spc 50 \
            --model_name ACGAN --hidden_size 64 --resize 32 --init_type normal --init_gain 0.02 \
            --algorithm fedcgan --eval_fraction 1 --eval_type both --eval_every 10 --eval_metrics acc1 fid \
            --R $((10 * $k)) --K $k --C $((10 / $k)) --E 1 --B 64 --max_workers $((100 / $k)) \
            --optimizer Adam --lr 0.001 --weight_decay 1e-4 --lr_decay 0.95 --lr_decay_step $k --criterion CrossEntropyLoss &
            sleep 1

            python3 main.py \
            --exp_name "CINIC10_FedCGAN_$s (K=$k; diri=1.00)" --seed $s --device cuda:1 \
            --dataset CINIC10 \
            --split_type diri --cncntrtn 1.00 --test_size 0.2 --spc 50 \
            --model_name ACGAN --hidden_size 64 --resize 32 --init_type normal --init_gain 0.02 \
            --algorithm fedcgan --eval_fraction 1 --eval_type both --eval_every 10 --eval_metrics acc1 fid \
            --R $((10 * $k)) --K $k --C $((10 / $k)) --E 1 --B 64 --max_workers $((100 / $k)) \
            --optimizer Adam --lr 0.001 --weight_decay 1e-4 --lr_decay 0.95 --lr_decay_step $k --criterion CrossEntropyLoss &
            sleep 1

            python3 main.py \
            --exp_name "MNIST_FedCGAN_$s (K=$k; diri=1.00)" --seed $s --device cuda:2 \
            --dataset MNIST \
            --split_type diri --cncntrtn 1.00 --test_size 0.2 --spc 50 \
            --model_name ACGAN --hidden_size 64 --resize 32 --init_type normal --init_gain 0.02 \
            --algorithm fedcgan --eval_fraction 1 --eval_type both --eval_every 10 --eval_metrics acc1 fid \
            --R $((10 * $k)) --K $k --C $((10 / $k)) --E 1 --B 64 --max_workers $((100 / $k)) \
            --optimizer Adam --lr 0.001 --weight_decay 1e-4 --lr_decay 0.95 --lr_decay_step $k --criterion CrossEntropyLoss
            sleep 1
        } &&
        echo "...done (K=$k; seed=$s)!"
    done
done