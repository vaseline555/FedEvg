#!/bin/sh

{
    python3 main.py \
    --exp_name "CIFAR10_FedEvg_0 (K=100; iid)" --seed 0 --device cuda:0 \
    --dataset CIFAR10 \
    --split_type iid --test_size 0.2 --spc 50 --bpr 10 \
    --model_name ResNet10 --hidden_size 64 --resize 32 --penult_spectral_norm \
    --algorithm fedevg --eval_fraction 1 --eval_type both --eval_every 10 --eval_metrics acc1 fid \
    --R 500 --K 100 --C 0.1 --E 1 --B 32 --server_beta 10. --server_beta_last 0.01 \
    --optimizer SGD --lr 0.01 --weight_decay 1e-4 --lr_decay 0.95 --lr_decay_step 50 --criterion CrossEntropyLoss &
    sleep 1

    python3 main.py \
    --exp_name "SVHN_FedEvg_0 (K=100; iid)" --seed 0 --device cuda:1 \
    --dataset SVHN \
    --split_type iid --test_size 0.2 --spc 50 --bpr 10 \
    --model_name ResNet10 --hidden_size 64 --resize 32 --penult_spectral_norm \
    --algorithm fedevg --eval_fraction 1 --eval_type both --eval_every 10 --eval_metrics acc1 fid \
    --R 500 --K 100 --C 0.1 --E 1 --B 32 --server_beta 10. --server_beta_last 0.01 \
    --optimizer SGD --lr 0.01 --weight_decay 1e-4 --lr_decay 0.95 --lr_decay_step 50 --criterion CrossEntropyLoss &

    python3 main.py \
    --exp_name "MNIST_FedEvg_0 (K=100; iid)" --seed 0 --device cuda:2 \
    --dataset MNIST \
    --split_type iid --test_size 0.2 --spc 50 --bpr 10 \
    --model_name ResNet10 --hidden_size 64 --resize 32 --penult_spectral_norm \
    --algorithm fedevg --eval_fraction 1 --eval_type both --eval_every 10 --eval_metrics acc1 fid \
    --R 500 --K 100 --C 0.1 --E 1 --B 32 --server_beta 10. --server_beta_last 0.01 \
    --optimizer SGD --lr 0.01 --weight_decay 1e-4 --lr_decay 0.95 --lr_decay_step 50 --criterion CrossEntropyLoss &
    sleep 1

    python3 main.py \
    --exp_name "CIFAR10_FedEvg_0 (K=100; diri=2)" --seed 0 --device cuda:0 \
    --dataset CIFAR10 \
    --split_type patho --mincls 2 --test_size 0.2 --spc 50 --bpr 10 \
    --model_name ResNet10 --hidden_size 64 --resize 32 --penult_spectral_norm \
    --algorithm fedevg --eval_fraction 1 --eval_type both --eval_every 10 --eval_metrics acc1 fid \
    --R 500 --K 100 --C 0.1 --E 1 --B 32 --server_beta 10. --server_beta_last 0.01 \
    --optimizer SGD --lr 0.01 --weight_decay 1e-4 --lr_decay 0.95 --lr_decay_step 50 --criterion CrossEntropyLoss &
    sleep 1

    python3 main.py \
    --exp_name "SVHN_FedEvg_0 (K=100; diri=2)" --seed 0 --device cuda:1 \
    --dataset SVHN \
    --split_type patho --mincls 2 --test_size 0.2 --spc 50 --bpr 10 \
    --model_name ResNet10 --hidden_size 64 --resize 32 --penult_spectral_norm \
    --algorithm fedevg --eval_fraction 1 --eval_type both --eval_every 10 --eval_metrics acc1 fid \
    --R 500 --K 100 --C 0.1 --E 1 --B 32 --server_beta 10. --server_beta_last 0.01 \
    --optimizer SGD --lr 0.01 --weight_decay 1e-4 --lr_decay 0.95 --lr_decay_step 50 --criterion CrossEntropyLoss &

    python3 main.py \
    --exp_name "MNIST_FedEvg_0 (K=100; diri=2)" --seed 0 --device cuda:2 \
    --dataset MNIST \
    --split_type patho --mincls 2 --test_size 0.2 --spc 50 --bpr 10 \
    --model_name ResNet10 --hidden_size 64 --resize 32 --penult_spectral_norm \
    --algorithm fedevg --eval_fraction 1 --eval_type both --eval_every 10 --eval_metrics acc1 fid \
    --R 500 --K 100 --C 0.1 --E 1 --B 32 --server_beta 10. --server_beta_last 0.01 \
    --optimizer SGD --lr 0.01 --weight_decay 1e-4 --lr_decay 0.95 --lr_decay_step 50 --criterion CrossEntropyLoss
} &&
sleep 1
echo "...done (K=100)!"

{
    python3 main.py \
    --exp_name "CIFAR10_FedEvg_0 (K=10; iid)" --seed 0 --device cuda:0 \
    --dataset CIFAR10 \
    --split_type iid --test_size 0.2 --spc 50 --bpr 10 \
    --model_name ResNet10 --hidden_size 64 --resize 32 --penult_spectral_norm \
    --algorithm fedevg --eval_fraction 1 --eval_type both --eval_every 10 --eval_metrics acc1 fid \
    --R 100 --K 10 --C 1 --E 1 --B 32 --server_beta 10. --server_beta_last 0.01 \
    --optimizer SGD --lr 0.01 --weight_decay 1e-4 --lr_decay 0.95 --lr_decay_step 10 --criterion CrossEntropyLoss &
    sleep 1

    python3 main.py \
    --exp_name "SVHN_FedEvg_0 (K=10; iid)" --seed 0 --device cuda:1 \
    --dataset SVHN \
    --split_type iid --test_size 0.2 --spc 50 --bpr 10 \
    --model_name ResNet10 --hidden_size 64 --resize 32 --penult_spectral_norm \
    --algorithm fedevg --eval_fraction 1 --eval_type both --eval_every 10 --eval_metrics acc1 fid \
    --R 100 --K 10 --C 1 --E 1 --B 32 --server_beta 10. --server_beta_last 0.01 \
    --optimizer SGD --lr 0.01 --weight_decay 1e-4 --lr_decay 0.95 --lr_decay_step 10 --criterion CrossEntropyLoss &

    python3 main.py \
    --exp_name "MNIST_FedEvg_0 (K=10; iid)" --seed 0 --device cuda:2 \
    --dataset MNIST \
    --split_type iid --test_size 0.2 --spc 50 --bpr 10 \
    --model_name ResNet10 --hidden_size 64 --resize 32 --penult_spectral_norm \
    --algorithm fedevg --eval_fraction 1 --eval_type both --eval_every 10 --eval_metrics acc1 fid \
    --R 100 --K 10 --C 1 --E 1 --B 32 --server_beta 10. --server_beta_last 0.01 \
    --optimizer SGD --lr 0.01 --weight_decay 1e-4 --lr_decay 0.95 --lr_decay_step 10 --criterion CrossEntropyLoss &
    sleep 1

    python3 main.py \
    --exp_name "CIFAR10_FedEvg_0 (K=10; diri=2)" --seed 0 --device cuda:0 \
    --dataset CIFAR10 \
    --split_type patho --mincls 2 --test_size 0.2 --spc 50 --bpr 10 \
    --model_name ResNet10 --hidden_size 64 --resize 32 --penult_spectral_norm \
    --algorithm fedevg --eval_fraction 1 --eval_type both --eval_every 10 --eval_metrics acc1 fid \
    --R 100 --K 10 --C 1 --E 1 --B 32 --server_beta 10. --server_beta_last 0.01 \
    --optimizer SGD --lr 0.01 --weight_decay 1e-4 --lr_decay 0.95 --lr_decay_step 10 --criterion CrossEntropyLoss &
    sleep 1

    python3 main.py \
    --exp_name "SVHN_FedEvg_0 (K=10; diri=2)" --seed 0 --device cuda:1 \
    --dataset SVHN \
    --split_type patho --mincls 2 --test_size 0.2 --spc 50 --bpr 10 \
    --model_name ResNet10 --hidden_size 64 --resize 32 --penult_spectral_norm \
    --algorithm fedevg --eval_fraction 1 --eval_type both --eval_every 10 --eval_metrics acc1 fid \
    --R 100 --K 10 --C 1 --E 1 --B 32 --server_beta 10. --server_beta_last 0.01 \
    --optimizer SGD --lr 0.01 --weight_decay 1e-4 --lr_decay 0.95 --lr_decay_step 10 --criterion CrossEntropyLoss &

    python3 main.py \
    --exp_name "MNIST_FedEvg_0 (K=10; diri=2)" --seed 0 --device cuda:2 \
    --dataset MNIST \
    --split_type patho --mincls 2 --test_size 0.2 --spc 50 --bpr 10 \
    --model_name ResNet10 --hidden_size 64 --resize 32 --penult_spectral_norm \
    --algorithm fedevg --eval_fraction 1 --eval_type both --eval_every 10 --eval_metrics acc1 fid \
    --R 100 --K 10 --C 1 --E 1 --B 32 --server_beta 10. --server_beta_last 0.01 \
    --optimizer SGD --lr 0.01 --weight_decay 1e-4 --lr_decay 0.95 --lr_decay_step 10 --criterion CrossEntropyLoss
} &&
sleep 1
echo "...done (K=10)!"

done