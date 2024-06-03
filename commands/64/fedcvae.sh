#!/bin/sh

for s in 1 2 3
do
    for k in 10 100 
    do
        echo "Start (K=$k; seed=$s)...!"
        {
            python3 main.py \
            --exp_name "DermaMNIST_FedCVAE_$s (K=$k; diri=0.01)" --seed $s --device cuda:0 \
            --dataset DermaMNIST \
            --split_type diri --cncntrtn 0.01 --test_size 0.2 --spc 10 \
            --model_name CVAE --hidden_size 64 --resize 64 \
            --algorithm fedcvae --eval_fraction 1 --eval_type both --eval_every $k --eval_metrics acc1 fid \
            --R $((10 * $k)) --K $k --C $((10 / $k)) --E 1 --B 64 --max_workers $((100 / $k)) \
            --optimizer Adam --lr 0.001 --weight_decay 1e-4 --lr_decay 0.95 --lr_decay_step $k --criterion MSELoss &
            sleep 1

            python3 main.py \
            --exp_name "OrganCMNIST_FedCVAE_$s (K=$k; diri=0.01)" --seed $s --device cuda:1 \
            --dataset OrganCMNIST \
            --split_type diri --cncntrtn 0.01 --test_size 0.2 --spc 10 \
            --model_name CVAE --hidden_size 64 --resize 64 \
            --algorithm fedcvae --eval_fraction 1 --eval_type both --eval_every $k --eval_metrics acc1 fid \
            --R $((10 * $k)) --K $k --C $((10 / $k)) --E 1 --B 64 --max_workers $((100 / $k)) \
            --optimizer Adam --lr 0.001 --weight_decay 1e-4 --lr_decay 0.95 --lr_decay_step $k --criterion MSELoss &
            sleep 1

            python3 main.py \
            --exp_name "BloodMNIST_FedCVAE_$s (K=$k; diri=0.01)" --seed $s --device cuda:2 \
            --dataset BloodMNIST \
            --split_type diri --cncntrtn 0.01 --test_size 0.2 --spc 10 \
            --model_name CVAE --hidden_size 64 --resize 64 \
            --algorithm fedcvae --eval_fraction 1 --eval_type both --eval_every $k --eval_metrics acc1 fid \
            --R $((10 * $k)) --K $k --C $((10 / $k)) --E 1 --B 64 --max_workers $((100 / $k)) \
            --optimizer Adam --lr 0.001 --weight_decay 1e-4 --lr_decay 0.95 --lr_decay_step $k --criterion MSELoss
            sleep 1
        } &&

        {
            python3 main.py \
            --exp_name "DermaMNIST_FedCVAE_$s (K=$k; diri=1.00)" --seed $s --device cuda:0 \
            --dataset DermaMNIST \
            --split_type diri --cncntrtn 1.00 --test_size 0.2 --spc 10 \
            --model_name CVAE --hidden_size 64 --resize 64 \
            --algorithm fedcvae --eval_fraction 1 --eval_type both --eval_every $k --eval_metrics acc1 fid \
            --R $((10 * $k)) --K $k --C $((10 / $k)) --E 1 --B 64 --max_workers $((100 / $k)) \
            --optimizer Adam --lr 0.001 --weight_decay 1e-4 --lr_decay 0.95 --lr_decay_step $k --criterion MSELoss &
            sleep 1

            python3 main.py \
            --exp_name "OrganCMNIST_FedCVAE_$s (K=$k; diri=1.00)" --seed $s --device cuda:1 \
            --dataset OrganCMNIST \
            --split_type diri --cncntrtn 1.00 --test_size 0.2 --spc 10 \
            --model_name CVAE --hidden_size 64 --resize 64 \
            --algorithm fedcvae --eval_fraction 1 --eval_type both --eval_every $k --eval_metrics acc1 fid \
            --R $((10 * $k)) --K $k --C $((10 / $k)) --E 1 --B 64 --max_workers $((100 / $k)) \
            --optimizer Adam --lr 0.001 --weight_decay 1e-4 --lr_decay 0.95 --lr_decay_step $k --criterion MSELoss &
            sleep 1

            python3 main.py \
            --exp_name "BloodMNIST_FedCVAE_$s (K=$k; diri=1.00)" --seed $s --device cuda:2 \
            --dataset BloodMNIST \
            --split_type diri --cncntrtn 1.00 --test_size 0.2 --spc 10 \
            --model_name CVAE --hidden_size 64 --resize 64 \
            --algorithm fedcvae --eval_fraction 1 --eval_type both --eval_every $k --eval_metrics acc1 fid \
            --R $((10 * $k)) --K $k --C $((10 / $k)) --E 1 --B 64 --max_workers $((100 / $k)) \
            --optimizer Adam --lr 0.001 --weight_decay 1e-4 --lr_decay 0.95 --lr_decay_step $k --criterion MSELoss
            sleep 1
        } &&
        echo "...done (K=$k; seed=$s)!"
    done
done