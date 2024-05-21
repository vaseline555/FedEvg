#!/bin/sh
# 60000 images * (1 - test_size) // K clients // B batch * (E * R) iters -> 60000 iters
# num params. (for communicaiton per client): 4,978,890 (ResNet10 - hidden=64) 

for s in 1 #2 3
do 
    {
    python3 main.py \
    --exp_name "MNIST_FedEvg_$s (K=10; alpha=0.01)" --seed $s --device cuda:0 \
    --dataset MNIST \
    --split_type diri --cncntrtn 0.01 --test_size 0.2 \
    --model_name ResNet10 --hidden_size 64 --resize 32 --penult_spectral_norm \
    --algorithm fedevg --eval_fraction 1 --eval_type both --eval_every 10 --eval_metrics acc1 fid \
    --R 100 --K 10 --C 1. --E 1 --B 64 --server_lr 10. \
    --optimizer Adam --lr 0.001 --lr_decay 0.999 --lr_decay_step 10 --criterion CrossEntropyLoss &
    sleep 1

    python3 main.py \
    --exp_name "MNIST_FedEvg_$s (K=20; alpha=0.01)" --seed $s --device cuda:1 \
    --dataset MNIST \
    --split_type diri --cncntrtn 0.01 --test_size 0.2 \
    --model_name ResNet10 --hidden_size 64 --resize 32 --penult_spectral_norm \
    --algorithm fedevg --eval_fraction 1 --eval_type both --eval_every 10 --eval_metrics acc1 fid \
    --R 100 --K 20 --C 0.5 --E 2 --B 64 --server_lr 10. \
    --optimizer Adam --lr 0.001 --lr_decay 0.999 --lr_decay_step 10 --criterion CrossEntropyLoss &
    sleep 1

    python3 main.py \
    --exp_name "MNIST_FedEvg_$s (K=50; alpha=0.01)" --seed $s --device cuda:2 \
    --dataset MNIST \
    --split_type diri --cncntrtn 0.01 --test_size 0.2 \
    --model_name ResNet10 --hidden_size 64 --resize 32 --penult_spectral_norm \
    --algorithm fedevg --eval_fraction 1 --eval_type both --eval_every 10 --eval_metrics acc1 fid \
    --R 100 --K 50 --C 0.2 --E 5 --B 64 --server_lr 10. \
    --optimizer Adam --lr 0.001 --lr_decay 0.999 --lr_decay_step 10 --criterion CrossEntropyLoss 
    } &&
    sleep 1

    {
    python3 main.py \
    --exp_name "MNIST_FedEvg_$s (K=10; alpha=1)" --seed $s --device cuda:0 \
    --dataset MNIST \
    --split_type diri --cncntrtn 1 --test_size 0.2 \
    --model_name ResNet10 --hidden_size 64 --resize 32 --penult_spectral_norm \
    --algorithm fedevg --eval_fraction 1 --eval_type both --eval_every 10 --eval_metrics acc1 fid \
    --R 100 --K 10 --C 1. --E 1 --B 64 --server_lr 10. \
    --optimizer Adam --lr 0.001 --lr_decay 0.999 --lr_decay_step 10 --criterion CrossEntropyLoss &
    sleep 1

    python3 main.py \
    --exp_name "MNIST_FedEvg_$s (K=20; alpha=1)" --seed $s --device cuda:1 \
    --dataset MNIST \
    --split_type diri --cncntrtn 1 --test_size 0.2 \
    --model_name ResNet10 --hidden_size 64 --resize 32 --penult_spectral_norm \
    --algorithm fedevg --eval_fraction 1 --eval_type both --eval_every 10 --eval_metrics acc1 fid \
    --R 100 --K 20 --C 0.5 --E 2 --B 64 --server_lr 10. \
    --optimizer Adam --lr 0.001 --lr_decay 0.999 --lr_decay_step 10 --criterion CrossEntropyLoss &
    sleep 1

    python3 main.py \
    --exp_name "MNIST_FedEvg_$s (K=50; alpha=1)" --seed $s --device cuda:2 \
    --dataset MNIST \
    --split_type diri --cncntrtn 1 --test_size 0.2 \
    --model_name ResNet10 --hidden_size 64 --resize 32 --penult_spectral_norm \
    --algorithm fedevg --eval_fraction 1 --eval_type both --eval_every 10 --eval_metrics acc1 fid \
    --R 100 --K 50 --C 0.2 --E 5 --B 64 --server_lr 10. \
    --optimizer Adam --lr 0.001 --lr_decay 0.999 --lr_decay_step 10 --criterion CrossEntropyLoss
    } &&
    sleep 1

    {
    python3 main.py \
    --exp_name "MNIST_FedEvg_$s (K=10; alpha=100)" --seed $s --device cuda:0 \
    --dataset MNIST \
    --split_type diri --cncntrtn 100 --test_size 0.2 \
    --model_name ResNet10 --hidden_size 64 --resize 32 --penult_spectral_norm \
    --algorithm fedevg --eval_fraction 1 --eval_type both --eval_every 10 --eval_metrics acc1 fid \
    --R 100 --K 10 --C 1. --E 1 --B 64 --server_lr 10. \
    --optimizer Adam --lr 0.001 --lr_decay 0.999 --lr_decay_step 10 --criterion CrossEntropyLoss &
    sleep 1

    python3 main.py \
    --exp_name "MNIST_FedEvg_$s (K=20; alpha=100)" --seed $s --device cuda:1 \
    --dataset MNIST \
    --split_type diri --cncntrtn 100 --test_size 0.2 \
    --model_name ResNet10 --hidden_size 64 --resize 32 --penult_spectral_norm \
    --algorithm fedevg --eval_fraction 1 --eval_type both --eval_every 10 --eval_metrics acc1 fid \
    --R 100 --K 20 --C 0.5 --E 2 --B 64 --server_lr 10. \
    --optimizer Adam --lr 0.001 --lr_decay 0.999 --lr_decay_step 10 --criterion CrossEntropyLoss &
    sleep 1

    python3 main.py \
    --exp_name "MNIST_FedEvg_$s (K=50; alpha=100)" --seed $s --device cuda:2 \
    --dataset MNIST \
    --split_type diri --cncntrtn 100 --test_size 0.2 \
    --model_name ResNet10 --hidden_size 64 --resize 32 --penult_spectral_norm \
    --algorithm fedevg --eval_fraction 1 --eval_type both --eval_every 10 --eval_metrics acc1 fid \
    --R 100 --K 50 --C 0.2 --E 5 --B 64 --server_lr 10. \
    --optimizer Adam --lr 0.001 --lr_decay 0.999 --lr_decay_step 10 --criterion CrossEntropyLoss 
    } &&
    sleep 1

    echo "...done!"
done