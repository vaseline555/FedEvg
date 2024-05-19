#!/bin/sh
# 60000 images * (1 - test_size) // K clients // B batches * (E * R) iterations -> 60000 iters
# num params. (for communicaiton per client): 1,924,747

for a in 0.01 1 100
do 
    python3 main.py \
    --exp_name "MNIST_FedCGAN_1 (K=10; alpha=$a)" --seed 1 --device cuda:0 \
    --dataset MNIST \
    --split_type diri --cncntrtn $a --test_size 0.2 --max_workers 4 \
    --model_name ACGAN --hidden_size 64 --resize 32 --init_type normal --init_gain 0.02 \
    --algorithm fedcgan --eval_fraction 1 --eval_type both --eval_every 20 --eval_metrics acc1 fid \
    --R 200 --K 10 --C 1. --E 1 --B 20 \
    --optimizer Adam --lr 0.001 --lr_decay 0.999 --lr_decay_step 10 --criterion CrossEntropyLoss &
    sleep 1

    python3 main.py \
    --exp_name "MNIST_FedCGAN_1 (K=20; alpha=$a)" --seed 1 --device cuda:1 \
    --dataset MNIST \
    --split_type diri --cncntrtn $a --test_size 0.2 --max_workers 4 \
    --model_name ACGAN --hidden_size 64 --resize 32 --init_type normal --init_gain 0.02 \
    --algorithm fedcgan --eval_fraction 1 --eval_type both --eval_every 40 --eval_metrics acc1 fid \
    --R 400 --K 20 --C 0.5 --E 1 --B 20 \
    --optimizer Adam --lr 0.001 --lr_decay 0.999 --lr_decay_step 20 --criterion CrossEntropyLoss &
    sleep 1

    python3 main.py \
    --exp_name "MNIST_FedCGAN_1 (K=50; alpha=$a)" --seed 1 --device cuda:2 \
    --dataset MNIST \
    --split_type diri --cncntrtn $a --test_size 0.2 --max_workers 4 \
    --model_name ACGAN --hidden_size 64 --resize 32 --init_type normal --init_gain 0.02 \
    --algorithm fedcgan --eval_fraction 1 --eval_type both --eval_every 100 --eval_metrics acc1 fid \
    --R 1000 --K 50 --C 0.2 --E 1 --B 20 \
    --optimizer Adam --lr 0.001 --lr_decay 0.999 --lr_decay_step 50 --criterion CrossEntropyLoss &
    sleep 1

    python3 main.py \
    --exp_name "MNIST_FedCGAN_1 (K=100; alpha=$a)" --seed 1 --device cuda:0 \
    --dataset MNIST \
    --split_type diri --cncntrtn $a --test_size 0.2 --max_workers 4 \
    --model_name ACGAN --hidden_size 64 --resize 32 --init_type normal --init_gain 0.02 \
    --algorithm fedcgan --eval_fraction 1 --eval_type both --eval_every 200 --eval_metrics acc1 fid \
    --R 2000 --K 100 --C 0.1 --E 1 --B 20 \
    --optimizer Adam --lr 0.001 --lr_decay 0.999 --lr_decay_step 100 --criterion CrossEntropyLoss &
    sleep 1

    python3 main.py \
    --exp_name "MNIST_FedCGAN_2 (K=10; alpha=$a)" --seed 2 --device cuda:1 \
    --dataset MNIST \
    --split_type diri --cncntrtn $a --test_size 0.2 --max_workers 4 \
    --model_name ACGAN --hidden_size 64 --resize 32 --init_type normal --init_gain 0.02 \
    --algorithm fedcgan --eval_fraction 1 --eval_type both --eval_every 20 --eval_metrics acc1 fid \
    --R 200 --K 10 --C 1. --E 1 --B 20 \
    --optimizer Adam --lr 0.001 --lr_decay 0.999 --lr_decay_step 10 --criterion CrossEntropyLoss &
    sleep 1

    python3 main.py \
    --exp_name "MNIST_FedCGAN_2 (K=20; alpha=$a)" --seed 2 --device cuda:2 \
    --dataset MNIST \
    --split_type diri --cncntrtn $a --test_size 0.2 --max_workers 4 \
    --model_name ACGAN --hidden_size 64 --resize 32 --init_type normal --init_gain 0.02 \
    --algorithm fedcgan --eval_fraction 1 --eval_type both --eval_every 40 --eval_metrics acc1 fid \
    --R 400 --K 20 --C 0.5 --E 1 --B 20 \
    --optimizer Adam --lr 0.001 --lr_decay 0.999 --lr_decay_step 20 --criterion CrossEntropyLoss &
    sleep 1

    python3 main.py \
    --exp_name "MNIST_FedCGAN_2 (K=50; alpha=$a)" --seed 2 --device cuda:0 \
    --dataset MNIST \
    --split_type diri --cncntrtn $a --test_size 0.2 --max_workers 4 \
    --model_name ACGAN --hidden_size 64 --resize 32 --init_type normal --init_gain 0.02 \
    --algorithm fedcgan --eval_fraction 1 --eval_type both --eval_every 100 --eval_metrics acc1 fid \
    --R 1000 --K 50 --C 0.2 --E 1 --B 20 \
    --optimizer Adam --lr 0.001 --lr_decay 0.999 --lr_decay_step 50 --criterion CrossEntropyLoss &
    sleep 1

    python3 main.py \
    --exp_name "MNIST_FedCGAN_2 (K=100; alpha=$a)" --seed 2 --device cuda:1 \
    --dataset MNIST \
    --split_type diri --cncntrtn $a --test_size 0.2 --max_workers 4 \
    --model_name ACGAN --hidden_size 64 --resize 32 --init_type normal --init_gain 0.02 \
    --algorithm fedcgan --eval_fraction 1 --eval_type both --eval_every 200 --eval_metrics acc1 fid \
    --R 2000 --K 100 --C 0.1 --E 1 --B 20 \
    --optimizer Adam --lr 0.001 --lr_decay 0.999 --lr_decay_step 100 --criterion CrossEntropyLoss &
    sleep 1

    python3 main.py \
    --exp_name "MNIST_FedCGAN_3 (K=10; alpha=$a)" --seed 3 --device cuda:2 \
    --dataset MNIST \
    --split_type diri --cncntrtn $a --test_size 0.2 --max_workers 4 \
    --model_name ACGAN --hidden_size 64 --resize 32 --init_type normal --init_gain 0.02 \
    --algorithm fedcgan --eval_fraction 1 --eval_type both --eval_every 20 --eval_metrics acc1 fid \
    --R 200 --K 10 --C 1. --E 1 --B 20 \
    --optimizer Adam --lr 0.001 --lr_decay 0.999 --lr_decay_step 10 --criterion CrossEntropyLoss &
    sleep 1

    python3 main.py \
    --exp_name "MNIST_FedCGAN_3 (K=20; alpha=$a)" --seed 3 --device cuda:0 \
    --dataset MNIST \
    --split_type diri --cncntrtn $a --test_size 0.2 --max_workers 4 \
    --model_name ACGAN --hidden_size 64 --resize 32 --init_type normal --init_gain 0.02 \
    --algorithm fedcgan --eval_fraction 1 --eval_type both --eval_every 40 --eval_metrics acc1 fid \
    --R 400 --K 20 --C 0.5 --E 1 --B 20 \
    --optimizer Adam --lr 0.001 --lr_decay 0.999 --lr_decay_step 20 --criterion CrossEntropyLoss &
    sleep 1

    python3 main.py \
    --exp_name "MNIST_FedCGAN_3 (K=50; alpha=$a)" --seed 3 --device cuda:1 \
    --dataset MNIST \
    --split_type diri --cncntrtn $a --test_size 0.2 --max_workers 4 \
    --model_name ACGAN --hidden_size 64 --resize 32 --init_type normal --init_gain 0.02 \
    --algorithm fedcgan --eval_fraction 1 --eval_type both --eval_every 100 --eval_metrics acc1 fid \
    --R 1000 --K 50 --C 0.2 --E 1 --B 20 \
    --optimizer Adam --lr 0.001 --lr_decay 0.999 --lr_decay_step 50 --criterion CrossEntropyLoss &
    sleep 1

    python3 main.py \
    --exp_name "MNIST_FedCGAN_3 (K=100; alpha=$a)" --seed 3 --device cuda:2 \
    --dataset MNIST \
    --split_type diri --cncntrtn $a --test_size 0.2 --max_workers 4 \
    --model_name ACGAN --hidden_size 64 --resize 32 --init_type normal --init_gain 0.02 \
    --algorithm fedcgan --eval_fraction 1 --eval_type both --eval_every 200 --eval_metrics acc1 fid \
    --R 2000 --K 100 --C 0.1 --E 1 --B 20 \
    --optimizer Adam --lr 0.001 --lr_decay 0.999 --lr_decay_step 100 --criterion CrossEntropyLoss &&
    sleep 1

    echo "...done alpha: $a!"
done