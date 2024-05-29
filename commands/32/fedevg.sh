#!/bin/sh

for s in 1 2 3
do
    for k in 10 20 50 
    do
        echo "Start (K=$k; seed=$s)...!"
        {
            python3 main.py \
            --exp_name "CIFAR10_FedEvg_$s (K=$k; patho=2)" --seed $s --device cuda:0 \
            --dataset CIFAR10 \
            --split_type patho --mincls 2 --test_size 0.2 --spc 10 \
            --model_name ResNet10 --hidden_size 64 --resize 32 --penult_spectral_norm \
            --algorithm fedevg --eval_fraction 1 --eval_type both --eval_every $k --eval_metrics acc1 fid \
            --R $(($k * 10)) --K $k --C $((5 / $k)) --E 1 --B 64 --max_workers $((200 / $k)) \
            --alpha 1.0 --sigma 0.01 --ld_steps 10 \
            --optimizer Adam --lr 0.001 --weight_decay 1e-4 --lr_decay 0.99 --lr_decay_step $k --criterion CrossEntropyLoss &
            sleep 1

            python3 main.py \
            --exp_name "SVHN_FedEvg_$s (K=$k; patho=2)" --seed $s --device cuda:1 \
            --dataset SVHN \
            --split_type patho --mincls 2 --test_size 0.2 --spc 10 \
            --model_name ResNet10 --hidden_size 64 --resize 32 --penult_spectral_norm \
            --algorithm fedevg --eval_fraction 1 --eval_type both --eval_every $k --eval_metrics acc1 fid \
            --R $(($k * 10)) --K $k --C $((5 / $k)) --E 1 --B 64 --max_workers $((200 / $k)) \
            --alpha 1.0 --sigma 0.01 --ld_steps 10 \
            --optimizer Adam --lr 0.001 --weight_decay 1e-4 --lr_decay 0.99 --lr_decay_step $k --criterion CrossEntropyLoss &
            sleep 1

            python3 main.py \
            --exp_name "MNIST_FedEvg_$s (K=$k; patho=2)" --seed $s --device cuda:2 \
            --dataset MNIST \
            --split_type patho --mincls 2 --test_size 0.2 --spc 10 \
            --model_name ResNet10 --hidden_size 64 --resize 32 --penult_spectral_norm \
            --algorithm fedevg --eval_fraction 1 --eval_type both --eval_every $k --eval_metrics acc1 fid \
            --R $(($k * 10)) --K $k --C $((5 / $k)) --E 1 --B 64 --max_workers $((200 / $k)) \
            --alpha 1.0 --sigma 0.01 --ld_steps 10 \
            --optimizer Adam --lr 0.001 --weight_decay 1e-4 --lr_decay 0.99 --lr_decay_step $k --criterion CrossEntropyLoss &
            sleep 1

            python3 main.py \
            --exp_name "CIFAR10_FedEvg_$s (K=$k; patho=5)" --seed $s --device cuda:0 \
            --dataset CIFAR10 \
            --split_type patho --mincls 5 --test_size 0.2 --spc 10 \
            --model_name ResNet10 --hidden_size 64 --resize 32 --penult_spectral_norm \
            --algorithm fedevg --eval_fraction 1 --eval_type both --eval_every $k --eval_metrics acc1 fid \
            --R $(($k * 10)) --K $k --C $((5 / $k)) --E 1 --B 64 --max_workers $((200 / $k)) \
            --alpha 1.0 --sigma 0.01 --ld_steps 10 \
            --optimizer Adam --lr 0.001 --weight_decay 1e-4 --lr_decay 0.99 --lr_decay_step $k --criterion CrossEntropyLoss &
            sleep 1

            python3 main.py \
            --exp_name "SVHN_FedEvg_$s (K=$k; patho=5)" --seed $s --device cuda:1 \
            --dataset SVHN \
            --split_type patho --mincls 5 --test_size 0.2 --spc 10 \
            --model_name ResNet10 --hidden_size 64 --resize 32 --penult_spectral_norm \
            --algorithm fedevg --eval_fraction 1 --eval_type both --eval_every $k --eval_metrics acc1 fid \
            --R $(($k * 10)) --K $k --C $((5 / $k)) --E 1 --B 64 --max_workers $((200 / $k)) \
            --alpha 1.0 --sigma 0.01 --ld_steps 10 \
            --optimizer Adam --lr 0.001 --weight_decay 1e-4 --lr_decay 0.99 --lr_decay_step $k --criterion CrossEntropyLoss &
            sleep 1

            python3 main.py \
            --exp_name "MNIST_FedEvg_$s (K=$k; patho=5)" --seed $s --device cuda:2 \
            --dataset MNIST \
            --split_type patho --mincls 5 --test_size 0.2 --spc 10 \
            --model_name ResNet10 --hidden_size 64 --resize 32 --penult_spectral_norm \
            --algorithm fedevg --eval_fraction 1 --eval_type both --eval_every $k --eval_metrics acc1 fid \
            --R $(($k * 10)) --K $k --C $((5 / $k)) --E 1 --B 64 --max_workers $((200 / $k)) \
            --alpha 1.0 --sigma 0.01 --ld_steps 10 \
            --optimizer Adam --lr 0.001 --weight_decay 1e-4 --lr_decay 0.99 --lr_decay_step $k --criterion CrossEntropyLoss
            sleep 1
        } &&
        echo "...done (K=$k; seed=$s)!"
    done
done