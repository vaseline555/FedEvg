#!/bin/sh

for s in 1 2 3
do
    for k in 10 50 100 
    do
        echo "Start (K=$k; seed=$s)...!"
        {
            python3 main.py \
            --exp_name "CIFAR10_FedCVAE_$s (K=$k; patho=2)" --seed $s --device cuda:0 \
            --dataset CIFAR10 \
            --split_type patho --mincls 2 --test_size 0.2 --spc 10 \
            --model_name CVAE --hidden_size 64 --resize 32 \
            --algorithm fedcvae --eval_fraction 1 --eval_type both --eval_every 1 --eval_metrics acc1 fid \
            --R 1 --K $k --C 1 --E $k --B 32 --max_workers 4 \
            --optimizer Adam --lr 0.001 --weight_decay 1e-4 --criterion MSELoss &
            sleep 1

            python3 main.py \
            --exp_name "SVHN_FedCVAE_$s (K=$k; patho=2)" --seed $s --device cuda:1 \
            --dataset SVHN \
            --split_type patho --mincls 2 --test_size 0.2 --spc 10 \
            --model_name CVAE --hidden_size 64 --resize 32 \
            --algorithm fedcvae --eval_fraction 1 --eval_type both --eval_every 1 --eval_metrics acc1 fid \
            --R 1 --K $k --C 1 --E $k --B 32 --max_workers 4 \
            --optimizer Adam --lr 0.001 --weight_decay 1e-4 --criterion MSELoss &
            sleep 1

            python3 main.py \
            --exp_name "MNIST_FedCVAE_$s (K=$k; patho=2)" --seed $s --device cuda:2 \
            --dataset MNIST \
            --split_type patho --mincls 2 --test_size 0.2 --spc 10 \
            --model_name CVAE --hidden_size 64 --resize 32 \
            --algorithm fedcvae --eval_fraction 1 --eval_type both --eval_every 1 --eval_metrics acc1 fid \
            --R 1 --K $k --C 1 --E $k --B 32 --max_workers 4 \
            --optimizer Adam --lr 0.001 --weight_decay 1e-4 --criterion MSELoss &
            sleep 1

            python3 main.py \
            --exp_name "CIFAR10_FedCVAE_$s (K=$k; patho=5)" --seed $s --device cuda:0 \
            --dataset CIFAR10 \
            --split_type patho --mincls 5 --test_size 0.2 --spc 10 \
            --model_name CVAE --hidden_size 64 --resize 32 \
            --algorithm fedcvae --eval_fraction 1 --eval_type both --eval_every 1 --eval_metrics acc1 fid \
            --R 1 --K $k --C 1 --E $k --B 32 --max_workers 4 \
            --optimizer Adam --lr 0.001 --weight_decay 1e-4 --criterion MSELoss &
            sleep 1

            python3 main.py \
            --exp_name "SVHN_FedCVAE_$s (K=$k; patho=5)" --seed $s --device cuda:1 \
            --dataset SVHN \
            --split_type patho --mincls 5 --test_size 0.2 --spc 10 \
            --model_name CVAE --hidden_size 64 --resize 32 \
            --algorithm fedcvae --eval_fraction 1 --eval_type both --eval_every 1 --eval_metrics acc1 fid \
            --R 1 --K $k --C 1 --E $k --B 32 --max_workers 4 \
            --optimizer Adam --lr 0.001 --weight_decay 1e-4 --criterion MSELoss &
            sleep 1

            python3 main.py \
            --exp_name "MNIST_FedCVAE_$s (K=$k; patho=5)" --seed $s --device cuda:2 \
            --dataset MNIST \
            --split_type patho --mincls 5 --test_size 0.2 --spc 10 \
            --model_name CVAE --hidden_size 64 --resize 32 \
            --algorithm fedcvae --eval_fraction 1 --eval_type both --eval_every 1 --eval_metrics acc1 fid \
            --R 1 --K $k --C 1 --E $k --B 32 --max_workers 4 \
            --optimizer Adam --lr 0.001 --weight_decay 1e-4 --criterion MSELoss
            sleep 1
        } &&
        echo "...done (K=$k; seed=$s)!"
    done
done