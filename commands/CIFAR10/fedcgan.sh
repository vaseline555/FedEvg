#!/bin/sh
# 60000 images * (1 - test_size) // K clients // B batches * (E * R) iterations -> 60000 iters
# num params. (for communicaiton per client): 1,924,747

{
python3 main.py \
--exp_name "CIFAR10_FedCGAN_1 (K=10; alpha=0.01)" --seed 1 --device cuda:0 \
--dataset CIFAR10 \
--split_type diri --cncntrtn 0.01 --test_size 0.2 --spc 20 \
--model_name ACGAN --hidden_size 64 --resize 32 --init_type normal --init_gain 0.02 \
--algorithm fedcgan --eval_fraction 1 --eval_type both --eval_every 10 --eval_metrics acc1 fid \
--R 100 --K 10 --C 1. --E 1 --B 64 \
--optimizer Adam --lr 0.001 --lr_decay 0.999 --lr_decay_step 10 --criterion CrossEntropyLoss &
sleep 1

python3 main.py \
--exp_name "CIFAR10_FedCGAN_1 (K=50; alpha=0.01)" --seed 1 --device cuda:1 \
--dataset CIFAR10 \
--split_type diri --cncntrtn 0.01 --test_size 0.2 --spc 20 \
--model_name ACGAN --hidden_size 64 --resize 32 --init_type normal --init_gain 0.02 \
--algorithm fedcgan --eval_fraction 1 --eval_type both --eval_every 50 --eval_metrics acc1 fid \
--R 500 --K 50 --C 0.2 --E 1 --B 64 \
--optimizer Adam --lr 0.001 --lr_decay 0.999 --lr_decay_step 50 --criterion CrossEntropyLoss &
sleep 1

python3 main.py \
--exp_name "CIFAR10_FedCGAN_1 (K=10; alpha=1)" --seed 1 --device cuda:2 \
--dataset CIFAR10 \
--split_type diri --cncntrtn 1 --test_size 0.2 --spc 20 \
--model_name ACGAN --hidden_size 64 --resize 32 --init_type normal --init_gain 0.02 \
--algorithm fedcgan --eval_fraction 1 --eval_type both --eval_every 10 --eval_metrics acc1 fid \
--R 100 --K 10 --C 1. --E 1 --B 64 \
--optimizer Adam --lr 0.001 --lr_decay 0.999 --lr_decay_step 10 --criterion CrossEntropyLoss &
sleep 1

python3 main.py \
--exp_name "CIFAR10_FedCGAN_1 (K=50; alpha=1)" --seed 1 --device cuda:0 \
--dataset CIFAR10 \
--split_type diri --cncntrtn 1 --test_size 0.2 --spc 20 \
--model_name ACGAN --hidden_size 64 --resize 32 --init_type normal --init_gain 0.02 \
--algorithm fedcgan --eval_fraction 1 --eval_type both --eval_every 50 --eval_metrics acc1 fid \
--R 500 --K 50 --C 0.2 --E 1 --B 64 \
--optimizer Adam --lr 0.001 --lr_decay 0.999 --lr_decay_step 50 --criterion CrossEntropyLoss &
sleep 1

python3 main.py \
--exp_name "CIFAR10_FedCGAN_2 (K=10; alpha=0.01)" --seed 2 --device cuda:1 \
--dataset CIFAR10 \
--split_type diri --cncntrtn 0.01 --test_size 0.2 --spc 20 \
--model_name ACGAN --hidden_size 64 --resize 32 --init_type normal --init_gain 0.02 \
--algorithm fedcgan --eval_fraction 1 --eval_type both --eval_every 10 --eval_metrics acc1 fid \
--R 100 --K 10 --C 1. --E 1 --B 64 \
--optimizer Adam --lr 0.001 --lr_decay 0.999 --lr_decay_step 10 --criterion CrossEntropyLoss &
sleep 1

python3 main.py \
--exp_name "CIFAR10_FedCGAN_2 (K=50; alpha=0.01)" --seed 2 --device cuda:2 \
--dataset CIFAR10 \
--split_type diri --cncntrtn 0.01 --test_size 0.2 --spc 20 \
--model_name ACGAN --hidden_size 64 --resize 32 --init_type normal --init_gain 0.02 \
--algorithm fedcgan --eval_fraction 1 --eval_type both --eval_every 50 --eval_metrics acc1 fid \
--R 500 --K 50 --C 0.2 --E 1 --B 64 \
--optimizer Adam --lr 0.001 --lr_decay 0.999 --lr_decay_step 50 --criterion CrossEntropyLoss
} &&
sleep 1

{
python3 main.py \
--exp_name "CIFAR10_FedCGAN_2 (K=10; alpha=1)" --seed 2 --device cuda:0 \
--dataset CIFAR10 \
--split_type diri --cncntrtn 1 --test_size 0.2 --spc 20 \
--model_name ACGAN --hidden_size 64 --resize 32 --init_type normal --init_gain 0.02 \
--algorithm fedcgan --eval_fraction 1 --eval_type both --eval_every 10 --eval_metrics acc1 fid \
--R 100 --K 10 --C 1. --E 1 --B 64 \
--optimizer Adam --lr 0.001 --lr_decay 0.999 --lr_decay_step 10 --criterion CrossEntropyLoss &
sleep 1

python3 main.py \
--exp_name "CIFAR10_FedCGAN_2 (K=50; alpha=1)" --seed 2 --device cuda:1 \
--dataset CIFAR10 \
--split_type diri --cncntrtn 1 --test_size 0.2 --spc 20 \
--model_name ACGAN --hidden_size 64 --resize 32 --init_type normal --init_gain 0.02 \
--algorithm fedcgan --eval_fraction 1 --eval_type both --eval_every 50 --eval_metrics acc1 fid \
--R 500 --K 50 --C 0.2 --E 1 --B 64 \
--optimizer Adam --lr 0.001 --lr_decay 0.999 --lr_decay_step 50 --criterion CrossEntropyLoss &
sleep 1

python3 main.py \
--exp_name "CIFAR10_FedCGAN_3 (K=10; alpha=0.01)" --seed 3 --device cuda:2 \
--dataset CIFAR10 \
--split_type diri --cncntrtn 0.01 --test_size 0.2 --spc 20 \
--model_name ACGAN --hidden_size 64 --resize 32 --init_type normal --init_gain 0.02 \
--algorithm fedcgan --eval_fraction 1 --eval_type both --eval_every 10 --eval_metrics acc1 fid \
--R 100 --K 10 --C 1. --E 1 --B 64 \
--optimizer Adam --lr 0.001 --lr_decay 0.999 --lr_decay_step 10 --criterion CrossEntropyLoss &
sleep 1

python3 main.py \
--exp_name "CIFAR10_FedCGAN_3 (K=50; alpha=0.01)" --seed 3 --device cuda:0 \
--dataset CIFAR10 \
--split_type diri --cncntrtn 0.01 --test_size 0.2 --spc 20 \
--model_name ACGAN --hidden_size 64 --resize 32 --init_type normal --init_gain 0.02 \
--algorithm fedcgan --eval_fraction 1 --eval_type both --eval_every 50 --eval_metrics acc1 fid \
--R 500 --K 50 --C 0.2 --E 1 --B 64 \
--optimizer Adam --lr 0.001 --lr_decay 0.999 --lr_decay_step 50 --criterion CrossEntropyLoss &
sleep 1

python3 main.py \
--exp_name "CIFAR10_FedCGAN_3 (K=10; alpha=1)" --seed 3 --device cuda:1 \
--dataset CIFAR10 \
--split_type diri --cncntrtn 1 --test_size 0.2 --spc 20 \
--model_name ACGAN --hidden_size 64 --resize 32 --init_type normal --init_gain 0.02 \
--algorithm fedcgan --eval_fraction 1 --eval_type both --eval_every 10 --eval_metrics acc1 fid \
--R 100 --K 10 --C 1. --E 1 --B 64 \
--optimizer Adam --lr 0.001 --lr_decay 0.999 --lr_decay_step 10 --criterion CrossEntropyLoss &
sleep 1

python3 main.py \
--exp_name "CIFAR10_FedCGAN_3 (K=50; alpha=1)" --seed 3 --device cuda:2 \
--dataset CIFAR10 \
--split_type diri --cncntrtn 1 --test_size 0.2 --spc 20 \
--model_name ACGAN --hidden_size 64 --resize 32 --init_type normal --init_gain 0.02 \
--algorithm fedcgan --eval_fraction 1 --eval_type both --eval_every 50 --eval_metrics acc1 fid \
--R 500 --K 50 --C 0.2 --E 1 --B 64 \
--optimizer Adam --lr 0.001 --lr_decay 0.999 --lr_decay_step 50 --criterion CrossEntropyLoss
} &&
sleep 1

echo "...done!"
done