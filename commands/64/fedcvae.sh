#!/bin/sh

for s in 1 2 3
do
    echo "Start (K=100)...!"
    {
        python3 main.py \
        --exp_name "DermaMNIST_FedCVAE_$s (K=100; diri=1.0)" --seed $s --device cuda:0 \
        --dataset DermaMNIST \
        --split_type diri --cncntrtn 1.0 --test_size 0.2 --spc 10 \
        --model_name CVAE --hidden_size 64 --resize 64 \
        --algorithm fedcvae --eval_fraction 1 --eval_type both --eval_every 10 --eval_metrics acc1 fid \
        --R 1 --K 100 --C 1 --E 5 --B 32 \
        --optimizer Adam --lr 0.001 --weight_decay 1e-4 --lr_decay 0.95 --lr_decay_step 50 --criterion MSELoss &
        sleep 1

        python3 main.py \
        --exp_name "OrganCMNIST_FedCVAE_$s (K=100; diri=1.0)" --seed $s --device cuda:1 \
        --dataset OrganCMNIST \
        --split_type diri --cncntrtn 1.0 --test_size 0.2 --spc 10 \
        --model_name CVAE --hidden_size 64 --resize 64 \
        --algorithm fedcvae --eval_fraction 1 --eval_type both --eval_every 10 --eval_metrics acc1 fid \
        --R 1 --K 100 --C 1 --E 5 --B 32 \
        --optimizer Adam --lr 0.001 --weight_decay 1e-4 --lr_decay 0.95 --lr_decay_step 50 --criterion MSELoss &
        sleep 1

        python3 main.py \
        --exp_name "BloddMNIST_FedCVAE_$s (K=100; diri=1.0)" --seed $s --device cuda:2 \
        --dataset BloddMNIST \
        --split_type diri --cncntrtn 1.0 --test_size 0.2 --spc 10 \
        --model_name CVAE --hidden_size 64 --resize 64 \
        --algorithm fedcvae --eval_fraction 1 --eval_type both --eval_every 10 --eval_metrics acc1 fid \
        --R 1 --K 100 --C 1 --E 5 --B 32 \
        --optimizer Adam --lr 0.001 --weight_decay 1e-4 --lr_decay 0.95 --lr_decay_step 50 --criterion MSELoss &
        sleep 1

        python3 main.py \
        --exp_name "DermaMNIST_FedCVAE_$s (K=100; diri=0.01)" --seed $s --device cuda:0 \
        --dataset DermaMNIST \
        --split_type diri --cncntrtn 0.01 --test_size 0.2 --spc 10 \
        --model_name CVAE --hidden_size 64 --resize 64 \
        --algorithm fedcvae --eval_fraction 1 --eval_type both --eval_every 10 --eval_metrics acc1 fid \
        --R 1 --K 100 --C 1 --E 5 --B 32 \
        --optimizer Adam --lr 0.001 --weight_decay 1e-4 --lr_decay 0.95 --lr_decay_step 50 --criterion MSELoss &
        sleep 1

        python3 main.py \
        --exp_name "OrganCMNIST_FedCVAE_$s (K=100; diri=0.01)" --seed $s --device cuda:1 \
        --dataset OrganCMNIST \
        --split_type diri --cncntrtn 0.01 --test_size 0.2 --spc 10 \
        --model_name CVAE --hidden_size 64 --resize 64 \
        --algorithm fedcvae --eval_fraction 1 --eval_type both --eval_every 10 --eval_metrics acc1 fid \
        --R 1 --K 100 --C 1 --E 5 --B 32 \
        --optimizer Adam --lr 0.001 --weight_decay 1e-4 --lr_decay 0.95 --lr_decay_step 50 --criterion MSELoss &
        sleep 1

        python3 main.py \
        --exp_name "BloddMNIST_FedCVAE_$s (K=100; diri=0.01)" --seed $s --device cuda:2 \
        --dataset BloddMNIST \
        --split_type diri --cncntrtn 0.01 --test_size 0.2 --spc 10 \
        --model_name CVAE --hidden_size 64 --resize 64 \
        --algorithm fedcvae --eval_fraction 1 --eval_type both --eval_every 10 --eval_metrics acc1 fid \
        --R 1 --K 100 --C 1 --E 5 --B 32 \
        --optimizer Adam --lr 0.001 --weight_decay 1e-4 --lr_decay 0.95 --lr_decay_step 50 --criterion MSELoss
        sleep 1
    } &&
    echo "...done (K=100)!"
    sleep 1

    echo "Start (K=10)...!"
    {
        python3 main.py \
        --exp_name "DermaMNIST_FedCVAE_$s (K=10; diri=1.0)" --seed $s --device cuda:0 \
        --dataset DermaMNIST \
        --split_type diri --cncntrtn 1.0 --test_size 0.2 --spc 10 \
        --model_name CVAE --hidden_size 64 --resize 64 \
        --algorithm fedcvae --eval_fraction 1 --eval_type both --eval_every 10 --eval_metrics acc1 fid \
        --R 1 --K 10 --C 1 --E 50 --B 32 \
        --optimizer Adam --lr 0.001 --weight_decay 1e-4 --lr_decay 0.95 --lr_decay_step 10 --criterion MSELoss &
        sleep 1

        python3 main.py \
        --exp_name "OrganCMNIST_FedCVAE_$s (K=10; diri=1.0)" --seed $s --device cuda:1 \
        --dataset OrganCMNIST \
        --split_type diri --cncntrtn 1.0 --test_size 0.2 --spc 10 \
        --model_name CVAE --hidden_size 64 --resize 64 \
        --algorithm fedcvae --eval_fraction 1 --eval_type both --eval_every 10 --eval_metrics acc1 fid \
        --R 1 --K 10 --C 1 --E 50 --B 32 \
        --optimizer Adam --lr 0.001 --weight_decay 1e-4 --lr_decay 0.95 --lr_decay_step 10 --criterion MSELoss &
        sleep 1

        python3 main.py \
        --exp_name "BloddMNIST_FedCVAE_$s (K=10; diri=1.0)" --seed $s --device cuda:2 \
        --dataset BloddMNIST \
        --split_type diri --cncntrtn 1.0 --test_size 0.2 --spc 10 \
        --model_name CVAE --hidden_size 64 --resize 64 \
        --algorithm fedcvae --eval_fraction 1 --eval_type both --eval_every 10 --eval_metrics acc1 fid \
        --R 1 --K 10 --C 1 --E 50 --B 32 \
        --optimizer Adam --lr 0.001 --weight_decay 1e-4 --lr_decay 0.95 --lr_decay_step 10 --criterion MSELoss &
        sleep 1

        python3 main.py \
        --exp_name "DermaMNIST_FedCVAE_$s (K=10; diri=0.01)" --seed $s --device cuda:0 \
        --dataset DermaMNIST \
        --split_type diri --cncntrtn 0.01 --test_size 0.2 --spc 10 \
        --model_name CVAE --hidden_size 64 --resize 64 \
        --algorithm fedcvae --eval_fraction 1 --eval_type both --eval_every 10 --eval_metrics acc1 fid \
        --R 1 --K 10 --C 1 --E 50 --B 32 \
        --optimizer Adam --lr 0.001 --weight_decay 1e-4 --lr_decay 0.95 --lr_decay_step 10 --criterion MSELoss &
        sleep 1

        python3 main.py \
        --exp_name "OrganCMNIST_FedCVAE_$s (K=10; diri=0.01)" --seed $s --device cuda:1 \
        --dataset OrganCMNIST \
        --split_type diri --cncntrtn 0.01 --test_size 0.2 --spc 10 \
        --model_name CVAE --hidden_size 64 --resize 64 \
        --algorithm fedcvae --eval_fraction 1 --eval_type both --eval_every 10 --eval_metrics acc1 fid \
        --R 1 --K 10 --C 1 --E 50 --B 32 \
        --optimizer Adam --lr 0.001 --weight_decay 1e-4 --lr_decay 0.95 --lr_decay_step 10 --criterion MSELoss &
        sleep 1

        python3 main.py \
        --exp_name "BloddMNIST_FedCVAE_$s (K=10; diri=0.01)" --seed $s --device cuda:2 \
        --dataset BloddMNIST \
        --split_type diri --cncntrtn 0.01 --test_size 0.2 --spc 10 \
        --model_name CVAE --hidden_size 64 --resize 64 \
        --algorithm fedcvae --eval_fraction 1 --eval_type both --eval_every 10 --eval_metrics acc1 fid \
        --R 1 --K 10 --C 1 --E 50 --B 32 \
        --optimizer Adam --lr 0.001 --weight_decay 1e-4 --lr_decay 0.95 --lr_decay_step 10 --criterion MSELoss
        sleep 1
    } &&
    echo "...done (K=10)!"
    sleep 1
done