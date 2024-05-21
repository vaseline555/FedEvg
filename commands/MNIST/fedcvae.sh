#!/bin/sh
# 60000 images * (1 - test_size) // K clients // B batches * E iterations (client-side)
# + (10 * num_classes * int(C * K) generated images // B bathces * E epochs) * 2 (server-side)
## -> 60000 iters
# num params. (for communicaiton per client): 836,353 (CVAE decoder)

for s in 1 #2 3
do 
    {
    python3 main.py \
    --exp_name "MNIST_FedCVAE_$s (K=10; alpha=0.01)" --seed $s --device cuda:0 \
    --dataset MNIST \
    --split_type diri --cncntrtn 0.01 --test_size 0.2 \
    --model_name CVAE --hidden_size 64 --resize 32 \
    --algorithm fedcvae --eval_fraction 1 --eval_type both --eval_every 1 --eval_metrics fid \
    --R 1 --K 10 --C 1 --E 10 --B 64 \
    --optimizer Adam --lr 0.001 --criterion MSELoss &
    sleep 1

    python3 main.py \
    --exp_name "MNIST_FedCVAE_$s (K=20; alpha=0.01)" --seed $s --device cuda:1 \
    --dataset MNIST \
    --split_type diri --cncntrtn 0.01 --test_size 0.2 \
    --model_name CVAE --hidden_size 64 --resize 32 \
    --algorithm fedcvae --eval_fraction 1 --eval_type both --eval_every 1 --eval_metrics fid \
    --R 1 --K 20 --C 1 --E 20 --B 64 \
    --optimizer Adam --lr 0.001 --criterion MSELoss &
    sleep 1

    python3 main.py \
    --exp_name "MNIST_FedCVAE_$s (K=50; alpha=0.01)" --seed $s --device cuda:2 \
    --dataset MNIST \
    --split_type diri --cncntrtn 0.01 --test_size 0.2 \
    --model_name CVAE --hidden_size 64 --resize 32 \
    --algorithm fedcvae --eval_fraction 1 --eval_type both --eval_every 1 --eval_metrics fid \
    --R 1 --K 50 --C 1 --E 50 --B 64 \
    --optimizer Adam --lr 0.001 --criterion MSELoss
    } &&
    sleep 1

    {
    python3 main.py \
    --exp_name "MNIST_FedCVAE_$s (K=10; alpha=1)" --seed $s --device cuda:0 \
    --dataset MNIST \
    --split_type diri --cncntrtn 1 --test_size 0.2 \
    --model_name CVAE --hidden_size 64 --resize 32 \
    --algorithm fedcvae --eval_fraction 1 --eval_type both --eval_every 1 --eval_metrics fid \
    --R 1 --K 10 --C 1 --E 10 --B 64 \
    --optimizer Adam --lr 0.001 --criterion MSELoss &
    sleep 1

    python3 main.py \
    --exp_name "MNIST_FedCVAE_$s (K=20; alpha=1)" --seed $s --device cuda:1 \
    --dataset MNIST \
    --split_type diri --cncntrtn 1 --test_size 0.2 \
    --model_name CVAE --hidden_size 64 --resize 32 \
    --algorithm fedcvae --eval_fraction 1 --eval_type both --eval_every 1 --eval_metrics fid \
    --R 1 --K 20 --C 1 --E 20 --B 64 \
    --optimizer Adam --lr 0.001 --criterion MSELoss &
    sleep 1

    python3 main.py \
    --exp_name "MNIST_FedCVAE_$s (K=50; alpha=1)" --seed $s --device cuda:2 \
    --dataset MNIST \
    --split_type diri --cncntrtn 1 --test_size 0.2 \
    --model_name CVAE --hidden_size 64 --resize 32 \
    --algorithm fedcvae --eval_fraction 1 --eval_type both --eval_every 1 --eval_metrics fid \
    --R 1 --K 50 --C 1 --E 50 --B 64 \
    --optimizer Adam --lr 0.001 --criterion MSELoss
    } &&
    sleep 1

    {
    python3 main.py \
    --exp_name "MNIST_FedCVAE_$s (K=10; alpha=100)" --seed $s --device cuda:0 \
    --dataset MNIST \
    --split_type diri --cncntrtn 100 --test_size 0.2 \
    --model_name CVAE --hidden_size 64 --resize 32 \
    --algorithm fedcvae --eval_fraction 1 --eval_type both --eval_every 1 --eval_metrics fid \
    --R 1 --K 10 --C 1 --E 10 --B 64 \
    --optimizer Adam --lr 0.001 --criterion MSELoss &
    sleep 1

    python3 main.py \
    --exp_name "MNIST_FedCVAE_$s (K=20; alpha=100)" --seed $s --device cuda:1 \
    --dataset MNIST \
    --split_type diri --cncntrtn 100 --test_size 0.2 \
    --model_name CVAE --hidden_size 64 --resize 32 \
    --algorithm fedcvae --eval_fraction 1 --eval_type both --eval_every 1 --eval_metrics fid \
    --R 1 --K 20 --C 1 --E 20 --B 64 \
    --optimizer Adam --lr 0.001 --criterion MSELoss &
    sleep 1

    python3 main.py \
    --exp_name "MNIST_FedCVAE_$s (K=50; alpha=100)" --seed $s --device cuda:2 \
    --dataset MNIST \
    --split_type diri --cncntrtn 100 --test_size 0.2 \
    --model_name CVAE --hidden_size 64 --resize 32 \
    --algorithm fedcvae --eval_fraction 1 --eval_type both --eval_every 1 --eval_metrics fid \
    --R 1 --K 50 --C 1 --E 50 --B 64 \
    --optimizer Adam --lr 0.001 --criterion MSELoss
    } &&
    sleep 1

    echo "...done alpha: 1!"
done