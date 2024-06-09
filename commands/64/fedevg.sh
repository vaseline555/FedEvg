#!/bin/sh

for s in 1 #2 3
do
    echo "Start seed=$s...!"
    {
        python3 main.py \
        --exp_name "DermaMNIST_FedEvg_$s (K=10; diri=0.01)" --seed $s --device cuda:0 \
        --dataset DermaMNIST \
        --split_type diri --cncntrtn 0.01 --test_size 0.2 --spc 10 --penult_spectral_norm \
        --model_name ResNet10 --hidden_size 64 --resize 64 --max_workers 10 \
        --algorithm fedevg --eval_fraction 1 --eval_type both --eval_every 50 --eval_metrics acc1 fid \
        --R 500 --K 10 --C 1. --E 1 --B 64 \
        --alpha 1. --sigma 0.01 --ld_steps 1 --ld_threshold 3. --cd_init pcd --server_beta 10. --server_beta_last 1. --ce_lambda 0.1 \
        --optimizer Adam --lr 0.001 --weight_decay 1e-4 --lr_decay 0.9 --lr_decay_step 50 --criterion CrossEntropyLoss &
        sleep 1

        python3 main.py \
        --exp_name "OrganCMNIST_FedEvg_$s (K=10; diri=0.01)" --seed $s --device cuda:1 \
        --dataset OrganCMNIST \
        --split_type diri --cncntrtn 0.01 --test_size 0.2 --spc 10 --penult_spectral_norm \
        --model_name ResNet10 --hidden_size 64 --resize 64 --max_workers 10 \
        --algorithm fedevg --eval_fraction 1 --eval_type both --eval_every 50 --eval_metrics acc1 fid \
        --R 500 --K 10 --C 1. --E 1 --B 64 \
        --alpha 1. --sigma 0.01 --ld_steps 1 --ld_threshold 3. --cd_init pcd --server_beta 10. --server_beta_last 1. --ce_lambda 0.1 \
        --optimizer Adam --lr 0.001 --weight_decay 1e-4 --lr_decay 0.9 --lr_decay_step 50 --criterion CrossEntropyLoss &
        sleep 1

        python3 main.py \
        --exp_name "BloodMNIST_FedEvg_$s (K=10; diri=0.01)" --seed $s --device cuda:2 \
        --dataset BloodMNIST \
        --split_type diri --cncntrtn 0.01 --test_size 0.2 --spc 10 --penult_spectral_norm \
        --model_name ResNet10 --hidden_size 64 --resize 64 --max_workers 10 \
        --algorithm fedevg --eval_fraction 1 --eval_type both --eval_every 50 --eval_metrics acc1 fid \
        --R 500 --K 10 --C 1. --E 1 --B 64 \
        --alpha 1. --sigma 0.01 --ld_steps 1 --ld_threshold 3. --cd_init pcd --server_beta 10. --server_beta_last 1. --ce_lambda 0.1 \
        --optimizer Adam --lr 0.001 --weight_decay 1e-4 --lr_decay 0.9 --lr_decay_step 50 --criterion CrossEntropyLoss &
        sleep 1

        python3 main.py \
        --exp_name "DermaMNIST_FedEvg_$s (K=10; diri=1.00)" --seed $s --device cuda:0 \
        --dataset DermaMNIST \
        --split_type diri --cncntrtn 1.00 --test_size 0.2 --spc 10 --penult_spectral_norm \
        --model_name ResNet10 --hidden_size 64 --resize 64 --max_workers 10 \
        --algorithm fedevg --eval_fraction 1 --eval_type both --eval_every 50 --eval_metrics acc1 fid \
        --R 500 --K 10 --C 1. --E 1 --B 64 \
        --alpha 1. --sigma 0.01 --ld_steps 1 --ld_threshold 3. --cd_init pcd --server_beta 10. --server_beta_last 1. --ce_lambda 0.1 \
        --optimizer Adam --lr 0.001 --weight_decay 1e-4 --lr_decay 0.9 --lr_decay_step 50 --criterion CrossEntropyLoss &
        sleep 1

        python3 main.py \
        --exp_name "OrganCMNIST_FedEvg_$s (K=10; diri=1.00)" --seed $s --device cuda:1 \
        --dataset OrganCMNIST \
        --split_type diri --cncntrtn 1.00 --test_size 0.2 --spc 10 --penult_spectral_norm \
        --model_name ResNet10 --hidden_size 64 --resize 64 --max_workers 10 \
        --algorithm fedevg --eval_fraction 1 --eval_type both --eval_every 50 --eval_metrics acc1 fid \
        --R 500 --K 10 --C 1. --E 1 --B 64 \
        --alpha 1. --sigma 0.01 --ld_steps 1 --ld_threshold 3. --cd_init pcd --server_beta 10. --server_beta_last 1. --ce_lambda 0.1 \
        --optimizer Adam --lr 0.001 --weight_decay 1e-4 --lr_decay 0.9 --lr_decay_step 50 --criterion CrossEntropyLoss &
        sleep 1

        python3 main.py \
        --exp_name "BloodMNIST_FedEvg_$s (K=10; diri=1.00)" --seed $s --device cuda:2 \
        --dataset BloodMNIST \
        --split_type diri --cncntrtn 1.00 --test_size 0.2 --spc 10 --penult_spectral_norm \
        --model_name ResNet10 --hidden_size 64 --resize 64 --max_workers 10 \
        --algorithm fedevg --eval_fraction 1 --eval_type both --eval_every 50 --eval_metrics acc1 fid \
        --R 500 --K 10 --C 1. --E 1 --B 64 \
        --alpha 1. --sigma 0.01 --ld_steps 1 --ld_threshold 3. --cd_init pcd --server_beta 10. --server_beta_last 1. --ce_lambda 0.1 \
        --optimizer Adam --lr 0.001 --weight_decay 1e-4 --lr_decay 0.9 --lr_decay_step 50 --criterion CrossEntropyLoss 
        sleep 1 
    } &&

    {
        python3 main.py \
        --exp_name "DermaMNIST_FedEvg_$s (K=100; diri=0.01)" --seed $s --device cuda:0 \
        --dataset DermaMNIST \
        --split_type diri --cncntrtn 0.01 --test_size 0.2 --spc 10 --penult_spectral_norm \
        --model_name ResNet10 --hidden_size 64 --resize 64 --max_workers 10 \
        --algorithm fedevg --eval_fraction 1 --eval_type both --eval_every 50 --eval_metrics acc1 fid \
        --R 500 --K 100 --C .1 --E 1 --B 64 \
        --alpha 1. --sigma 0.01 --ld_steps 1 --ld_threshold 3. --cd_init pcd --server_beta 10. --server_beta_last 1. --ce_lambda 0.1 \
        --optimizer Adam --lr 0.001 --weight_decay 1e-4 --lr_decay 0.9 --lr_decay_step 50 --criterion CrossEntropyLoss &
        sleep 1

        python3 main.py \
        --exp_name "OrganCMNIST_FedEvg_$s (K=100; diri=0.01)" --seed $s --device cuda:1 \
        --dataset OrganCMNIST \
        --split_type diri --cncntrtn 0.01 --test_size 0.2 --spc 10 --penult_spectral_norm \
        --model_name ResNet10 --hidden_size 64 --resize 64 --max_workers 10 \
        --algorithm fedevg --eval_fraction 1 --eval_type both --eval_every 50 --eval_metrics acc1 fid \
        --R 500 --K 100 --C .1 --E 1 --B 64 \
        --alpha 1. --sigma 0.01 --ld_steps 1 --ld_threshold 3. --cd_init pcd --server_beta 10. --server_beta_last 1. --ce_lambda 0.1 \
        --optimizer Adam --lr 0.001 --weight_decay 1e-4 --lr_decay 0.9 --lr_decay_step 50 --criterion CrossEntropyLoss &
        sleep 1

        python3 main.py \
        --exp_name "BloodMNIST_FedEvg_$s (K=100; diri=0.01)" --seed $s --device cuda:2 \
        --dataset BloodMNIST \
        --split_type diri --cncntrtn 0.01 --test_size 0.2 --spc 10 --penult_spectral_norm \
        --model_name ResNet10 --hidden_size 64 --resize 64 --max_workers 10 \
        --algorithm fedevg --eval_fraction 1 --eval_type both --eval_every 50 --eval_metrics acc1 fid \
        --R 500 --K 100 --C .1 --E 1 --B 64 \
        --alpha 1. --sigma 0.01 --ld_steps 1 --ld_threshold 3. --cd_init pcd --server_beta 10. --server_beta_last 1. --ce_lambda 0.1 \
        --optimizer Adam --lr 0.001 --weight_decay 1e-4 --lr_decay 0.9 --lr_decay_step 50 --criterion CrossEntropyLoss &
        sleep 1

        python3 main.py \
        --exp_name "DermaMNIST_FedEvg_$s (K=100; diri=1.00)" --seed $s --device cuda:0 \
        --dataset DermaMNIST \
        --split_type diri --cncntrtn 1.00 --test_size 0.2 --spc 10 --penult_spectral_norm \
        --model_name ResNet10 --hidden_size 64 --resize 64 --max_workers 10 \
        --algorithm fedevg --eval_fraction 1 --eval_type both --eval_every 50 --eval_metrics acc1 fid \
        --R 500 --K 100 --C .1 --E 1 --B 64 \
        --alpha 1. --sigma 0.01 --ld_steps 1 --ld_threshold 3. --cd_init pcd --server_beta 10. --server_beta_last 1. --ce_lambda 0.1 \
        --optimizer Adam --lr 0.001 --weight_decay 1e-4 --lr_decay 0.9 --lr_decay_step 50 --criterion CrossEntropyLoss &
        sleep 1

        python3 main.py \
        --exp_name "OrganCMNIST_FedEvg_$s (K=100; diri=1.00)" --seed $s --device cuda:1 \
        --dataset OrganCMNIST \
        --split_type diri --cncntrtn 1.00 --test_size 0.2 --spc 10 --penult_spectral_norm \
        --model_name ResNet10 --hidden_size 64 --resize 64 --max_workers 10 \
        --algorithm fedevg --eval_fraction 1 --eval_type both --eval_every 50 --eval_metrics acc1 fid \
        --R 500 --K 100 --C .1 --E 1 --B 64 \
        --alpha 1. --sigma 0.01 --ld_steps 1 --ld_threshold 3. --cd_init pcd --server_beta 10. --server_beta_last 1. --ce_lambda 0.1 \
        --optimizer Adam --lr 0.001 --weight_decay 1e-4 --lr_decay 0.9 --lr_decay_step 50 --criterion CrossEntropyLoss &
        sleep 1

        python3 main.py \
        --exp_name "BloodMNIST_FedEvg_$s (K=100; diri=1.00)" --seed $s --device cuda:2 \
        --dataset BloodMNIST \
        --split_type diri --cncntrtn 1.00 --test_size 0.2 --spc 10 --penult_spectral_norm \
        --model_name ResNet10 --hidden_size 64 --resize 64 --max_workers 10 \
        --algorithm fedevg --eval_fraction 1 --eval_type both --eval_every 50 --eval_metrics acc1 fid \
        --R 500 --K 100 --C .1 --E 1 --B 64 \
        --alpha 1. --sigma 0.01 --ld_steps 1 --ld_threshold 3. --cd_init pcd --server_beta 10. --server_beta_last 1. --ce_lambda 0.1 \
        --optimizer Adam --lr 0.001 --weight_decay 1e-4 --lr_decay 0.9 --lr_decay_step 50 --criterion CrossEntropyLoss
        sleep 1
    } &&
    echo "...done seed=$s!"
done