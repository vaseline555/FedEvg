#!/bin/sh
# 60000 images * (1 - test_size) // K clients // B batch * (E * R) iters -> 40000 iters
# num params. (for communicaiton per client): 4,978,890 (ResNet10 - hidden=64) 

for s in 1 #2 3
do 
    {
    python3 main.py \
    --exp_name "MNIST_FedDM_$s (K=10; alpha=0.01)" --seed $s --device cuda:0 \
    --dataset MNIST \
    --split_type diri --cncntrtn 0.01 --test_size 0.2 \
    --model_name ResNet10 --hidden_size 64 --resize 32 \
    --algorithm feddm --eval_fraction 1 --eval_type both --eval_every 20 --eval_metrics acc1 fid \
    --R 200 --K 10 --C 1. --E 10 --B 256 \
    --optimizer SGD --lr 1.0 --momentum 0.5 --lr_decay 0.999 --lr_decay_step 20 --criterion CrossEntropyLoss &
    sleep 1

    python3 main.py \
    --exp_name "MNIST_FedDM_$s (K=20; alpha=0.01)" --seed $s --device cuda:1 \
    --dataset MNIST \
    --split_type diri --cncntrtn 0.01 --test_size 0.2 \
    --model_name ResNet10 --hidden_size 64 --resize 32 \
    --algorithm feddm --eval_fraction 1 --eval_type both --eval_every 20 --eval_metrics acc1 fid \
    --R 200 --K 20 --C 0.5 --E 20 --B 256 \
    --optimizer SGD --lr 1.0 --momentum 0.5 --lr_decay 0.999 --lr_decay_step 20 --criterion CrossEntropyLoss &
    sleep 1

    python3 main.py \
    --exp_name "MNIST_FedDM_$s (K=50; alpha=0.01)" --seed $s --device cuda:2 \
    --dataset MNIST \
    --split_type diri --cncntrtn 0.01 --test_size 0.2 \
    --model_name ResNet10 --hidden_size 64 --resize 32 \
    --algorithm feddm --eval_fraction 1 --eval_type both --eval_every 20 --eval_metrics acc1 fid \
    --R 200 --K 50 --C 0.2 --E 50 --B 256 \
    --optimizer SGD --lr 1.0 --momentum 0.5 --lr_decay 0.999 --lr_decay_step 20 --criterion CrossEntropyLoss 
    } &&
    sleep 1

    {
    python3 main.py \
    --exp_name "MNIST_FedDM_$s (K=10; alpha=1)" --seed $s --device cuda:0 \
    --dataset MNIST \
    --split_type diri --cncntrtn 1 --test_size 0.2 \
    --model_name ResNet10 --hidden_size 64 --resize 32 \
    --algorithm feddm --eval_fraction 1 --eval_type both --eval_every 20 --eval_metrics acc1 fid \
    --R 200 --K 10 --C 1. --E 10 --B 256 \
    --optimizer SGD --lr 1.0 --momentum 0.5 --lr_decay 0.999 --lr_decay_step 20 --criterion CrossEntropyLoss &
    sleep 1

    python3 main.py \
    --exp_name "MNIST_FedDM_$s (K=20; alpha=1)" --seed $s --device cuda:1 \
    --dataset MNIST \
    --split_type diri --cncntrtn 1 --test_size 0.2 \
    --model_name ResNet10 --hidden_size 64 --resize 32 \
    --algorithm feddm --eval_fraction 1 --eval_type both --eval_every 20 --eval_metrics acc1 fid \
    --R 200 --K 20 --C 0.5 --E 20 --B 256 \
    --optimizer SGD --lr 1.0 --momentum 0.5 --lr_decay 0.999 --lr_decay_step 20 --criterion CrossEntropyLoss &
    sleep 1

    python3 main.py \
    --exp_name "MNIST_FedDM_$s (K=50; alpha=1)" --seed $s --device cuda:2 \
    --dataset MNIST \
    --split_type diri --cncntrtn 1 --test_size 0.2 \
    --model_name ResNet10 --hidden_size 64 --resize 32 \
    --algorithm feddm --eval_fraction 1 --eval_type both --eval_every 20 --eval_metrics acc1 fid \
    --R 200 --K 50 --C 0.2 --E 50 --B 256 \
    --optimizer SGD --lr 1.0 --momentum 0.5 --lr_decay 0.999 --lr_decay_step 20 --criterion CrossEntropyLoss
    } &&
    sleep 1

    {
    python3 main.py \
    --exp_name "MNIST_FedDM_$s (K=10; alpha=200)" --seed $s --device cuda:0 \
    --dataset MNIST \
    --split_type diri --cncntrtn 200 --test_size 0.2 \
    --model_name ResNet10 --hidden_size 64 --resize 32 \
    --algorithm feddm --eval_fraction 1 --eval_type both --eval_every 20 --eval_metrics acc1 fid \
    --R 200 --K 10 --C 1. --E 10 --B 256 \
    --optimizer SGD --lr 1.0 --momentum 0.5 --lr_decay 0.999 --lr_decay_step 20 --criterion CrossEntropyLoss &
    sleep 1

    python3 main.py \
    --exp_name "MNIST_FedDM_$s (K=20; alpha=200)" --seed $s --device cuda:1 \
    --dataset MNIST \
    --split_type diri --cncntrtn 200 --test_size 0.2 \
    --model_name ResNet10 --hidden_size 64 --resize 32 \
    --algorithm feddm --eval_fraction 1 --eval_type both --eval_every 20 --eval_metrics acc1 fid \
    --R 200 --K 20 --C 0.5 --E 20 --B 256 \
    --optimizer SGD --lr 1.0 --momentum 0.5 --lr_decay 0.999 --lr_decay_step 20 --criterion CrossEntropyLoss &
    sleep 1

    python3 main.py \
    --exp_name "MNIST_FedDM_$s (K=50; alpha=200)" --seed $s --device cuda:2 \
    --dataset MNIST \
    --split_type diri --cncntrtn 200 --test_size 0.2 \
    --model_name ResNet10 --hidden_size 64 --resize 32 \
    --algorithm feddm --eval_fraction 1 --eval_type both --eval_every 20 --eval_metrics acc1 fid \
    --R 200 --K 50 --C 0.2 --E 50 --B 256 \
    --optimizer SGD --lr 1.0 --momentum 0.5 --lr_decay 0.999 --lr_decay_step 20 --criterion CrossEntropyLoss 
    } &&
    sleep 1

    echo "...done!"
done