#!/bin/sh
# 60000 images * (1 - test_size) // K clients // B batch * (E * R) iters -> 60000 iters
# num params. (for communicaiton per client): 50 (args.spc) * 10 (args.num_classes) * (1 * 32 * 32) = 512,000

{
python3 main.py \
--exp_name "CIFAR10_FedEvg_1 (K=10; alpha=0.01)" --seed 1 --device cuda:0 \
--dataset CIFAR10 \
--split_type diri --cncntrtn 0.01 --test_size 0.2 --spc 20 \
--model_name ResNet10 --hidden_size 64 --resize 32 \
--algorithm fedevg --eval_fraction 1 --eval_type both --eval_every 10 --eval_metrics acc1 fid \
--R 100 --K 10 --C 1. --E 1 --B 32 --server_lr 10. \
--optimizer SGD --lr 1.0 --weight_decay 1e-4 --lr_decay 0.95 --lr_decay_step 10 --criterion CrossEntropyLoss &
sleep 1

python3 main.py \
--exp_name "CIFAR10_FedEvg_1 (K=50; alpha=0.01)" --seed 1 --device cuda:1 \
--dataset CIFAR10 \
--split_type diri --cncntrtn 0.01 --test_size 0.2 --spc 20 \
--model_name ResNet10 --hidden_size 64 --resize 32 \
--algorithm fedevg --eval_fraction 1 --eval_type both --eval_every 10 --eval_metrics acc1 fid \
--R 100 --K 50 --C 0.2 --E 10 --B 32 --server_lr 10. \
--optimizer SGD --lr 1.0 --weight_decay 1e-4 --lr_decay 0.95 --lr_decay_step 10 --criterion CrossEntropyLoss &
sleep 1

python3 main.py \
--exp_name "CIFAR10_FedEvg_1 (K=10; alpha=1)" --seed 1 --device cuda:2 \
--dataset CIFAR10 \
--split_type diri --cncntrtn 1 --test_size 0.2 --spc 20 \
--model_name ResNet10 --hidden_size 64 --resize 32 \
--algorithm fedevg --eval_fraction 1 --eval_type both --eval_every 10 --eval_metrics acc1 fid \
--R 100 --K 10 --C 1. --E 1 --B 32 --server_lr 10. \
--optimizer SGD --lr 1.0 --weight_decay 1e-4 --lr_decay 0.95 --lr_decay_step 10 --criterion CrossEntropyLoss &
sleep 1

python3 main.py \
--exp_name "CIFAR10_FedEvg_1 (K=50; alpha=1)" --seed 1 --device cuda:0 \
--dataset CIFAR10 \
--split_type diri --cncntrtn 1 --test_size 0.2 --spc 20 \
--model_name ResNet10 --hidden_size 64 --resize 32 \
--algorithm fedevg --eval_fraction 1 --eval_type both --eval_every 10 --eval_metrics acc1 fid \
--R 100 --K 50 --C 0.2 --E 10 --B 32 --server_lr 10. \
--optimizer SGD --lr 1.0 --weight_decay 1e-4 --lr_decay 0.95 --lr_decay_step 10 --criterion CrossEntropyLoss &
sleep 1

python3 main.py \
--exp_name "CIFAR10_FedEvg_2 (K=10; alpha=0.01)" --seed 2 --device cuda:1 \
--dataset CIFAR10 \
--split_type diri --cncntrtn 1 --test_size 0.2 --spc 20 \
--model_name ResNet10 --hidden_size 64 --resize 32 \
--algorithm fedevg --eval_fraction 1 --eval_type both --eval_every 10 --eval_metrics acc1 fid \
--R 100 --K 10 --C 1. --E 1 --B 32 --server_lr 10. \
--optimizer SGD --lr 1.0 --weight_decay 1e-4 --lr_decay 0.95 --lr_decay_step 10 --criterion CrossEntropyLoss &
sleep 1

python3 main.py \
--exp_name "CIFAR10_FedEvg_2 (K=50; alpha=0.01)" --seed 2 --device cuda:2 \
--dataset CIFAR10 \
--split_type diri --cncntrtn 1 --test_size 0.2 --spc 20 \
--model_name ResNet10 --hidden_size 64 --resize 32 \
--algorithm fedevg --eval_fraction 1 --eval_type both --eval_every 10 --eval_metrics acc1 fid \
--R 100 --K 50 --C 0.2 --E 10 --B 32 --server_lr 10. \
--optimizer SGD --lr 1.0 --weight_decay 1e-4 --lr_decay 0.95 --lr_decay_step 10 --criterion CrossEntropyLoss 
} &&
sleep 1

{
python3 main.py \
--exp_name "CIFAR10_FedEvg_2 (K=10; alpha=1)" --seed 2 --device cuda:0 \
--dataset CIFAR10 \
--split_type diri --cncntrtn 1 --test_size 0.2 --spc 20 \
--model_name ResNet10 --hidden_size 64 --resize 32 \
--algorithm fedevg --eval_fraction 1 --eval_type both --eval_every 10 --eval_metrics acc1 fid \
--R 100 --K 10 --C 1. --E 1 --B 32 --server_lr 10. \
--optimizer SGD --lr 1.0 --weight_decay 1e-4 --lr_decay 0.95 --lr_decay_step 10 --criterion CrossEntropyLoss &
sleep 1

python3 main.py \
--exp_name "CIFAR10_FedEvg_2 (K=50; alpha=1)" --seed 2 --device cuda:1 \
--dataset CIFAR10 \
--split_type diri --cncntrtn 1 --test_size 0.2 --spc 20 \
--model_name ResNet10 --hidden_size 64 --resize 32 \
--algorithm fedevg --eval_fraction 1 --eval_type both --eval_every 10 --eval_metrics acc1 fid \
--R 100 --K 50 --C 0.2 --E 10 --B 32 --server_lr 10. \
--optimizer SGD --lr 1.0 --weight_decay 1e-4 --lr_decay 0.95 --lr_decay_step 10 --criterion CrossEntropyLoss &
sleep 1

python3 main.py \
--exp_name "CIFAR10_FedEvg_3 (K=10; alpha=0.01)" --seed 3 --device cuda:2 \
--dataset CIFAR10 \
--split_type diri --cncntrtn 1 --test_size 0.2 --spc 20 \
--model_name ResNet10 --hidden_size 64 --resize 32 \
--algorithm fedevg --eval_fraction 1 --eval_type both --eval_every 10 --eval_metrics acc1 fid \
--R 100 --K 10 --C 1. --E 1 --B 32 --server_lr 10. \
--optimizer SGD --lr 1.0 --weight_decay 1e-4 --lr_decay 0.95 --lr_decay_step 10 --criterion CrossEntropyLoss &
sleep 1

python3 main.py \
--exp_name "CIFAR10_FedEvg_3 (K=50; alpha=0.01)" --seed 3 --device cuda:0 \
--dataset CIFAR10 \
--split_type diri --cncntrtn 1 --test_size 0.2 --spc 20 \
--model_name ResNet10 --hidden_size 64 --resize 32 \
--algorithm fedevg --eval_fraction 1 --eval_type both --eval_every 10 --eval_metrics acc1 fid \
--R 100 --K 50 --C 0.2 --E 10 --B 32 --server_lr 10. \
--optimizer SGD --lr 1.0 --weight_decay 1e-4 --lr_decay 0.95 --lr_decay_step 10 --criterion CrossEntropyLoss &
sleep 1

python3 main.py \
--exp_name "CIFAR10_FedEvg_3 (K=10; alpha=1)" --seed 3 --device cuda:1 \
--dataset CIFAR10 \
--split_type diri --cncntrtn 1 --test_size 0.2 --spc 20 \
--model_name ResNet10 --hidden_size 64 --resize 32 \
--algorithm fedevg --eval_fraction 1 --eval_type both --eval_every 10 --eval_metrics acc1 fid \
--R 100 --K 10 --C 1. --E 1 --B 32 --server_lr 10. \
--optimizer SGD --lr 1.0 --weight_decay 1e-4 --lr_decay 0.95 --lr_decay_step 10 --criterion CrossEntropyLoss &
sleep 1

python3 main.py \
--exp_name "CIFAR10_FedEvg_3 (K=50; alpha=1)" --seed 3 --device cuda:2 \
--dataset CIFAR10 \
--split_type diri --cncntrtn 1 --test_size 0.2 --spc 20 \
--model_name ResNet10 --hidden_size 64 --resize 32 \
--algorithm fedevg --eval_fraction 1 --eval_type both --eval_every 10 --eval_metrics acc1 fid \
--R 100 --K 50 --C 0.2 --E 10 --B 32 --server_lr 10. \
--optimizer SGD --lr 1.0 --weight_decay 1e-4 --lr_decay 0.95 --lr_decay_step 10 --criterion CrossEntropyLoss 
} &&
sleep 1

echo "...done!"
done