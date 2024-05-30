#!/bin/sh

for s in 1 2 3
do
    for k in 10 100
    do
        echo "Start (K=$k; seed=$s)...!"
        {
            python3 main.py \
            --exp_name "DermaMNIST_FedEvg_$s (K=$k; diri=0.01)" --seed $s --device cuda:0 \
            --dataset DermaMNIST \
            --split_type diri --cncntrtn 0.01 --test_size 0.2 --bpr 10 --spc 50 --penult_spectral_norm \
            --model_name ResNet10 --hidden_size 64 --resize 64 \
            --algorithm fedevg --eval_fraction 1 --eval_type both --eval_every $k --eval_metrics acc1 fid \
            --R $((10 * $k)) --K $k --C $((10 / $k)) --E 1 --B 64 --max_workers $((100 / $k)) \
            --alpha 1.0 --sigma 0.01 --ld_steps 10 --server_beta 10. --server_beta_last 1. --ce_lambda 0.1 \
            --optimizer Adam --lr 0.001 --weight_decay 1e-4 --lr_decay 0.95 --lr_decay_step $k --criterion CrossEntropyLoss &
            sleep 1

            python3 main.py \
            --exp_name "OrganCMNIST_FedEvg_$s (K=$k; diri=0.01)" --seed $s --device cuda:1 \
            --dataset OrganCMNIST \
            --split_type diri --cncntrtn 0.01 --test_size 0.2 --bpr 10 --spc 50 --penult_spectral_norm \
            --model_name ResNet10 --hidden_size 64 --resize 64 \
            --algorithm fedevg --eval_fraction 1 --eval_type both --eval_every $k --eval_metrics acc1 fid \
            --R $((10 * $k)) --K $k --C $((10 / $k)) --E 1 --B 64 --max_workers $((100 / $k)) \
            --alpha 1.0 --sigma 0.01 --ld_steps 10 --server_beta 10. --server_beta_last 1. --ce_lambda 0.1 \
            --optimizer Adam --lr 0.001 --weight_decay 1e-4 --lr_decay 0.95 --lr_decay_step $k --criterion CrossEntropyLoss &
            sleep 1

            python3 main.py \
            --exp_name "BloodMNIST_FedEvg_$s (K=$k; diri=0.01)" --seed $s --device cuda:2 \
            --dataset BloodMNIST \
            --split_type diri --cncntrtn 0.01 --test_size 0.2 --bpr 10 --spc 50 --penult_spectral_norm \
            --model_name ResNet10 --hidden_size 64 --resize 64 \
            --algorithm fedevg --eval_fraction 1 --eval_type both --eval_every $k --eval_metrics acc1 fid \
            --R $((10 * $k)) --K $k --C $((10 / $k)) --E 1 --B 64 --max_workers $((100 / $k)) \
            --alpha 1.0 --sigma 0.01 --ld_steps 10 --server_beta 10. --server_beta_last 1. --ce_lambda 0.1 \
            --optimizer Adam --lr 0.001 --weight_decay 1e-4 --lr_decay 0.95 --lr_decay_step $k --criterion CrossEntropyLoss &
            sleep 1

            python3 main.py \
            --exp_name "DermaMNIST_FedEvg_$s (K=$k; diri=1.00)" --seed $s --device cuda:0 \
            --dataset DermaMNIST \
            --split_type diri --cncntrtn 1.00 --test_size 0.2 --bpr 10 --spc 50 --penult_spectral_norm \
            --model_name ResNet10 --hidden_size 64 --resize 64 \
            --algorithm fedevg --eval_fraction 1 --eval_type both --eval_every $k --eval_metrics acc1 fid \
            --R $((10 * $k)) --K $k --C $((10 / $k)) --E 1 --B 64 --max_workers $((100 / $k)) \
            --alpha 1.0 --sigma 0.01 --ld_steps 10 --server_beta 10. --server_beta_last 1. --ce_lambda 0.1 \
            --optimizer Adam --lr 0.001 --weight_decay 1e-4 --lr_decay 0.95 --lr_decay_step $k --criterion CrossEntropyLoss &
            sleep 1

            python3 main.py \
            --exp_name "OrganCMNIST_FedEvg_$s (K=$k; diri=1.00)" --seed $s --device cuda:1 \
            --dataset OrganCMNIST \
            --split_type diri --cncntrtn 1.00 --test_size 0.2 --bpr 10 --spc 50 --penult_spectral_norm \
            --model_name ResNet10 --hidden_size 64 --resize 64 \
            --algorithm fedevg --eval_fraction 1 --eval_type both --eval_every $k --eval_metrics acc1 fid \
            --R $((10 * $k)) --K $k --C $((10 / $k)) --E 1 --B 64 --max_workers $((100 / $k)) \
            --alpha 1.0 --sigma 0.01 --ld_steps 10 --server_beta 10. --server_beta_last 1. --ce_lambda 0.1 \
            --optimizer Adam --lr 0.001 --weight_decay 1e-4 --lr_decay 0.95 --lr_decay_step $k --criterion CrossEntropyLoss &
            sleep 1

            python3 main.py \
            --exp_name "BloodMNIST_FedEvg_$s (K=$k; diri=1.00)" --seed $s --device cuda:2 \
            --dataset BloodMNIST \
            --split_type diri --cncntrtn 1.00 --test_size 0.2 --bpr 10 --spc 50 --penult_spectral_norm \
            --model_name ResNet10 --hidden_size 64 --resize 64 \
            --algorithm fedevg --eval_fraction 1 --eval_type both --eval_every $k --eval_metrics acc1 fid \
            --R $((10 * $k)) --K $k --C $((10 / $k)) --E 1 --B 64 --max_workers $((100 / $k)) \
            --alpha 1.0 --sigma 0.01 --ld_steps 10 --server_beta 10. --server_beta_last 1. --ce_lambda 0.1 \
            --optimizer Adam --lr 0.001 --weight_decay 1e-4 --lr_decay 0.95 --lr_decay_step $k --criterion CrossEntropyLoss
            sleep 1
        } &&
        echo "...done (K=$k; seed=$s)!"
    done
done