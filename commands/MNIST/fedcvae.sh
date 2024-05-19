#!/bin/sh
# 60000 images * (1 - test_size) // K clients // B batches * E iterations (client-side)
# + (10000 generated images // B bathces * E epochs) * 2 (server-side)
## -> 60000 iters
# num params. (for communicaiton per client): 836,353 (CVAE decoder)

for a in 0.01 1 100
do 
    python3 main.py \
    --exp_name "MNIST_FedCVAE_1 (K=10; alpha=$a)" --seed 1 --device cuda:0 \
    --dataset MNIST \
    --split_type diri --cncntrtn $a --test_size 0.2 --max_workers 4 \
    --model_name CVAE --hidden_size 64 --resize 32 \
    --algorithm fedcvae --eval_fraction 1 --eval_type both --eval_every 1 --eval_metrics fid \
    --R 1 --K 10 --C 1 --E 150 --B 64 \
    --optimizer Adam --lr 0.001 --criterion MSELoss &
    sleep 1

    python3 main.py \
    --exp_name "MNIST_FedCVAE_1 (K=20; alpha=$a)" --seed 1 --device cuda:1 \
    --dataset MNIST \
    --split_type diri --cncntrtn $a --test_size 0.2 --max_workers 4 \
    --model_name CVAE --hidden_size 64 --resize 32 \
    --algorithm fedcvae --eval_fraction 1 --eval_type both --eval_every 1 --eval_metrics fid \
    --R 1 --K 20 --C 1 --E 160 --B 64 \
    --optimizer Adam --lr 0.001 --criterion MSELoss &
    sleep 1

    python3 main.py \
    --exp_name "MNIST_FedCVAE_1 (K=50; alpha=$a)" --seed 1 --device cuda:2 \
    --dataset MNIST \
    --split_type diri --cncntrtn $a --test_size 0.2 --max_workers 4 \
    --model_name CVAE --hidden_size 64 --resize 32 \
    --algorithm fedcvae --eval_fraction 1 --eval_type both --eval_every 1 --eval_metrics fid \
    --R 1 --K 50 --C 1 --E 180 --B 64 \
    --optimizer Adam --lr 0.001 --criterion MSELoss &
    sleep 1

    python3 main.py \
    --exp_name "MNIST_FedCVAE_1 (K=100; alpha=$a)" --seed 1 --device cuda:0 \
    --dataset MNIST \
    --split_type diri --cncntrtn $a --test_size 0.2 --max_workers 4 \
    --model_name CVAE --hidden_size 64 --resize 32 \
    --algorithm fedcvae --eval_fraction 1 --eval_type both --eval_every 1 --eval_metrics fid \
    --R 1 --K 100 --C 1 --E 185 --B 64 \
    --optimizer Adam --lr 0.001 --criterion MSELoss &
    sleep 1

    python3 main.py \
    --exp_name "MNIST_FedCVAE_2 (K=10; alpha=$a)" --seed 2 --device cuda:1 \
    --dataset MNIST \
    --split_type diri --cncntrtn $a --test_size 0.2 --max_workers 4 \
    --model_name CVAE --hidden_size 64 --resize 32 \
    --algorithm fedcvae --eval_fraction 1 --eval_type both --eval_every 1 --eval_metrics fid \
    --R 1 --K 10 --C 1 --E 160 --B 64 \
    --optimizer Adam --lr 0.001 --criterion MSELoss &
    sleep 1

    python3 main.py \
    --exp_name "MNIST_FedCVAE_2 (K=20; alpha=$a)" --seed 2 --device cuda:2 \
    --dataset MNIST \
    --split_type diri --cncntrtn $a --test_size 0.2 --max_workers 4 \
    --model_name CVAE --hidden_size 64 --resize 32 \
    --algorithm fedcvae --eval_fraction 1 --eval_type both --eval_every 1 --eval_metrics fid \
    --R 1 --K 20 --C 1 --E 100 --B 64 \
    --optimizer Adam --lr 0.001 --criterion MSELoss &
    sleep 1

    python3 main.py \
    --exp_name "MNIST_FedCVAE_2 (K=50; alpha=$a)" --seed 2 --device cuda:0 \
    --dataset MNIST \
    --split_type diri --cncntrtn $a --test_size 0.2 --max_workers 4 \
    --model_name CVAE --hidden_size 64 --resize 32 \
    --algorithm fedcvae --eval_fraction 1 --eval_type both --eval_every 1 --eval_metrics fid \
    --R 1 --K 50 --C 1 --E 180 --B 64 \
    --optimizer Adam --lr 0.001 --criterion MSELoss &
    sleep 1

    python3 main.py \
    --exp_name "MNIST_FedCVAE_2 (K=100; alpha=$a)" --seed 2 --device cuda:1 \
    --dataset MNIST \
    --split_type diri --cncntrtn $a --test_size 0.2 --max_workers 4 \
    --model_name CVAE --hidden_size 64 --resize 32 \
    --algorithm fedcvae --eval_fraction 1 --eval_type both --eval_every 1 --eval_metrics fid \
    --R 1 --K 100 --C 1 --E 185 --B 64 \
    --optimizer Adam --lr 0.001 --criterion MSELoss &
    sleep 1

    python3 main.py \
    --exp_name "MNIST_FedCVAE_3 (K=10; alpha=$a)" --seed 3 --device cuda:2 \
    --dataset MNIST \
    --split_type diri --cncntrtn $a --test_size 0.2 --max_workers 4 \
    --model_name CVAE --hidden_size 64 --resize 32 \
    --algorithm fedcvae --eval_fraction 1 --eval_type both --eval_every 1 --eval_metrics fid \
    --R 1 --K 10 --C 1 --E 160 --B 64 \
    --optimizer Adam --lr 0.001 --criterion MSELoss &
    sleep 1

    python3 main.py \
    --exp_name "MNIST_FedCVAE_3 (K=20; alpha=$a)" --seed 3 --device cuda:0 \
    --dataset MNIST \
    --split_type diri --cncntrtn $a --test_size 0.2 --max_workers 4 \
    --model_name CVAE --hidden_size 64 --resize 32 \
    --algorithm fedcvae --eval_fraction 1 --eval_type both --eval_every 1 --eval_metrics fid \
    --R 1 --K 20 --C 1 --E 100 --B 64 \
    --optimizer Adam --lr 0.001 --criterion MSELoss &
    sleep 1

    python3 main.py \
    --exp_name "MNIST_FedCVAE_3 (K=50; alpha=$a)" --seed 3 --device cuda:1 \
    --dataset MNIST \
    --split_type diri --cncntrtn $a --test_size 0.2 --max_workers 4 \
    --model_name CVAE --hidden_size 64 --resize 32 \
    --algorithm fedcvae --eval_fraction 1 --eval_type both --eval_every 1 --eval_metrics fid \
    --R 1 --K 50 --C 1 --E 180 --B 64 \
    --optimizer Adam --lr 0.001 --criterion MSELoss &
    sleep 1

    python3 main.py \
    --exp_name "MNIST_FedCVAE_3 (K=100; alpha=$a)" --seed 3 --device cuda:2 \
    --dataset MNIST \
    --split_type diri --cncntrtn $a --test_size 0.2 --max_workers 4 \
    --model_name CVAE --hidden_size 64 --resize 32 \
    --algorithm fedcvae --eval_fraction 1 --eval_type both --eval_every 1 --eval_metrics fid \
    --R 1 --K 100 --C 1 --E 185 --B 64 \
    --optimizer Adam --lr 0.001 --criterion MSELoss &&
    sleep 1

    echo "...done alpha: $a!"
done