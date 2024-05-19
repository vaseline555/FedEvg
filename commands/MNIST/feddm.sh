#!/bin/sh
# 60000 images * (1 - test_size) // K clients // B batches * E * R iterations  (client-side)
# + (10000 generated images // B bathces * 20 epochs) (server-side)
## -> 60000 iters
# num params. (for communicaiton per client): 4,978,890 (ResNet10 - hidden=64) 

for a in 0.01 1 100
do 
    python3 main.py \
    --exp_name "MNIST_FedDM_1 (K=10; alpha=$a)" --seed 1 --device cuda:0 \
    --dataset MNIST \
    --split_type diri --cncntrtn $a --test_size 0.2 --max_workers 4 \
    --model_name ResNet10 --hidden_size 64 --resize 32 \
    --algorithm feddm --eval_fraction 1 --eval_type both --eval_every 20 --eval_metrics acc1 fid \
    --R 200 --K 10 --C 1. --E 3 --B 64 \
    --optimizer SGD --lr 1.0 --momentum 0.5 --criterion CrossEntropyLoss &
    sleep 1

    python3 main.py \
    --exp_name "MNIST_FedDM_1 (K=20; alpha=$a)" --seed 1 --device cuda:1 \
    --dataset MNIST \
    --split_type diri --cncntrtn $a --test_size 0.2 --max_workers 4 \
    --model_name ResNet10 --hidden_size 64 --resize 32 \
    --algorithm feddm --eval_fraction 1 --eval_type both --eval_every 40 --eval_metrics acc1 fid \
    --R 400 --K 20 --C 0.5 --E 3 --B 64 \
    --optimizer SGD --lr 1.0 --momentum 0.5 --criterion CrossEntropyLoss &
    sleep 1

    python3 main.py \
    --exp_name "MNIST_FedDM_1 (K=50; alpha=$a)" --seed 1 --device cuda:2 \
    --dataset MNIST \
    --split_type diri --cncntrtn $a --test_size 0.2 --max_workers 4 \
    --model_name ResNet10 --hidden_size 64 --resize 32 \
    --algorithm feddm --eval_fraction 1 --eval_type both --eval_every 100 --eval_metrics acc1 fid \
    --R 1000 --K 50 --C 0.2 --E 3 --B 64 \
    --optimizer SGD --lr 1.0 --momentum 0.5 --criterion CrossEntropyLoss &
    sleep 1

    python3 main.py \
    --exp_name "MNIST_FedDM_1 (K=100; alpha=$a)" --seed 1 --device cuda:0 \
    --dataset MNIST \
    --split_type diri --cncntrtn $a --test_size 0.2 --max_workers 4 \
    --model_name ResNet10 --hidden_size 64 --resize 32 \
    --algorithm feddm --eval_fraction 1 --eval_type both --eval_every 200 --eval_metrics acc1 fid \
    --R 2000 --K 100 --C 0.1 --E 3 --B 64 \
    --optimizer SGD --lr 1.0 --momentum 0.5 --criterion CrossEntropyLoss &
    sleep 1

    python3 main.py \
    --exp_name "MNIST_FedDM_2 (K=10; alpha=$a)" --seed 2 --device cuda:1 \
    --dataset MNIST \
    --split_type diri --cncntrtn $a --test_size 0.2 --max_workers 4 \
    --model_name ResNet10 --hidden_size 64 --resize 32 \
    --algorithm feddm --eval_fraction 1 --eval_type both --eval_every 20 --eval_metrics acc1 fid \
    --R 200 --K 10 --C 1. --E 3 --B 64 \
    --optimizer SGD --lr 1.0 --momentum 0.5 --criterion CrossEntropyLoss &
    sleep 1

    python3 main.py \
    --exp_name "MNIST_FedDM_2 (K=20; alpha=$a)" --seed 2 --device cuda:2 \
    --dataset MNIST \
    --split_type diri --cncntrtn $a --test_size 0.2 --max_workers 4 \
    --model_name ResNet10 --hidden_size 64 --resize 32 \
    --algorithm feddm --eval_fraction 1 --eval_type both --eval_every 40 --eval_metrics acc1 fid \
    --R 400 --K 20 --C 0.5 --E 3 --B 64 \
    --optimizer SGD --lr 1.0 --momentum 0.5 --criterion CrossEntropyLoss &
    sleep 1

    python3 main.py \
    --exp_name "MNIST_FedDM_2 (K=50; alpha=$a)" --seed 2 --device cuda:0 \
    --dataset MNIST \
    --split_type diri --cncntrtn $a --test_size 0.2 --max_workers 4 \
    --model_name ResNet10 --hidden_size 64 --resize 32 \
    --algorithm feddm --eval_fraction 1 --eval_type both --eval_every 100 --eval_metrics acc1 fid \
    --R 1000 --K 50 --C 0.2 --E 3 --B 64 \
    --optimizer SGD --lr 1.0 --momentum 0.5 --criterion CrossEntropyLoss &
    sleep 1

    python3 main.py \
    --exp_name "MNIST_FedDM_2 (K=100; alpha=$a)" --seed 2 --device cuda:1 \
    --dataset MNIST \
    --split_type diri --cncntrtn $a --test_size 0.2 --max_workers 4 \
    --model_name ResNet10 --hidden_size 64 --resize 32 \
    --algorithm feddm --eval_fraction 1 --eval_type both --eval_every 200 --eval_metrics acc1 fid \
    --R 2000 --K 100 --C 0.1 --E 3 --B 64 \
    --optimizer SGD --lr 1.0 --momentum 0.5 --criterion CrossEntropyLoss &
    sleep 1

    python3 main.py \
    --exp_name "MNIST_FedDM_3 (K=10; alpha=$a)" --seed 3 --device cuda:2 \
    --dataset MNIST \
    --split_type diri --cncntrtn $a --test_size 0.2 --max_workers 4 \
    --model_name ResNet10 --hidden_size 64 --resize 32 \
    --algorithm feddm --eval_fraction 1 --eval_type both --eval_every 20 --eval_metrics acc1 fid \
    --R 200 --K 10 --C 1. --E 3 --B 64 \
    --optimizer SGD --lr 1.0 --momentum 0.5 --criterion CrossEntropyLoss &
    sleep 1

    python3 main.py \
    --exp_name "MNIST_FedDM_3 (K=20; alpha=$a)" --seed 3 --device cuda:0 \
    --dataset MNIST \
    --split_type diri --cncntrtn $a --test_size 0.2 --max_workers 4 \
    --model_name ResNet10 --hidden_size 64 --resize 32 \
    --algorithm feddm --eval_fraction 1 --eval_type both --eval_every 40 --eval_metrics acc1 fid \
    --R 400 --K 20 --C 0.5 --E 3 --B 64 \
    --optimizer SGD --lr 1.0 --momentum 0.5 --criterion CrossEntropyLoss &
    sleep 1

    python3 main.py \
    --exp_name "MNIST_FedDM_3 (K=50; alpha=$a)" --seed 3 --device cuda:1 \
    --dataset MNIST \
    --split_type diri --cncntrtn $a --test_size 0.2 --max_workers 4 \
    --model_name ResNet10 --hidden_size 64 --resize 32 \
    --algorithm feddm --eval_fraction 1 --eval_type both --eval_every 100 --eval_metrics acc1 fid \
    --R 1000 --K 50 --C 0.2 --E 3 --B 64 \
    --optimizer SGD --lr 1.0 --momentum 0.5 --criterion CrossEntropyLoss &
    sleep 1

    python3 main.py \
    --exp_name "MNIST_FedDM_3 (K=100; alpha=$a)" --seed 3 --device cuda:2 \
    --dataset MNIST \
    --split_type diri --cncntrtn $a --test_size 0.2 --max_workers 4 \
    --model_name ResNet10 --hidden_size 64 --resize 32 \
    --algorithm feddm --eval_fraction 1 --eval_type both --eval_every 200 --eval_metrics acc1 fid \
    --R 2000 --K 100 --C 0.1 --E 3 --B 64 \
    --optimizer SGD --lr 1.0 --momentum 0.5 --criterion CrossEntropyLoss &&
    sleep 1

    echo "...done alpha: $a!"
done