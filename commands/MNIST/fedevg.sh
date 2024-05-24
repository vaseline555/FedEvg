#!/bin/sh
# 60000 images * (1 - test_size) // K clients // B batch * (E * R) iters -> 60000 iters
# num params. (for communicaiton per client): 50 (args.spc) * 10 (args.num_classes) * (1 * 32 * 32) = 512,000

{
python3 main.py \
--exp_name "MNIST_FedEvg_1 (K=10; alpha=0.01)" --seed 1 --device cuda:0 \
--dataset MNIST \
--split_type diri --cncntrtn 0.01 --test_size 0.2 --spc 50 \
--model_name ResNet10 --hidden_size 64 --resize 32 \
--algorithm fedevg --eval_fraction 1 --eval_type both --eval_every 10 --eval_metrics acc1 fid \
--R 100 --K 10 --C 1. --E 1 --B 64 --server_lr 10. \
--optimizer Adam --lr 0.0001 --weight_decay 1e-4 --lr_decay 0.999 --lr_decay_step 10 --criterion CrossEntropyLoss &
sleep 1

python3 main.py \
--exp_name "MNIST_FedEvg_1 (K=50; alpha=0.01)" --seed 1 --device cuda:1 \
--dataset MNIST \
--split_type diri --cncntrtn 0.01 --test_size 0.2 --spc 50 \
--model_name ResNet10 --hidden_size 64 --resize 32 \
--algorithm fedevg --eval_fraction 1 --eval_type both --eval_every 50 --eval_metrics acc1 fid \
--R 500 --K 50 --C 0.2 --E 1 --B 64 --server_lr 10. \
--optimizer Adam --lr 0.0001 --weight_decay 1e-4 --lr_decay 0.999 --lr_decay_step 50 --criterion CrossEntropyLoss &
sleep 1

python3 main.py \
--exp_name "MNIST_FedEvg_1 (K=10; alpha=1)" --seed 1 --device cuda:2 \
--dataset MNIST \
--split_type diri --cncntrtn 1 --test_size 0.2 --spc 50 \
--model_name ResNet10 --hidden_size 64 --resize 32 \
--algorithm fedevg --eval_fraction 1 --eval_type both --eval_every 10 --eval_metrics acc1 fid \
--R 100 --K 10 --C 1. --E 1 --B 64 --server_lr 10. \
--optimizer Adam --lr 0.0001 --weight_decay 1e-4 --lr_decay 0.999 --lr_decay_step 10 --criterion CrossEntropyLoss &
sleep 1

python3 main.py \
--exp_name "MNIST_FedEvg_1 (K=50; alpha=1)" --seed 1 --device cuda:0 \
--dataset MNIST \
--split_type diri --cncntrtn 1 --test_size 0.2 --spc 50 \
--model_name ResNet10 --hidden_size 64 --resize 32 \
--algorithm fedevg --eval_fraction 1 --eval_type both --eval_every 50 --eval_metrics acc1 fid \
--R 500 --K 50 --C 0.2 --E 1 --B 64 --server_lr 10. \
--optimizer Adam --lr 0.0001 --weight_decay 1e-4 --lr_decay 0.999 --lr_decay_step 50 --criterion CrossEntropyLoss &
sleep 1

python3 main.py \
--exp_name "MNIST_FedEvg_2 (K=10; alpha=0.01)" --seed 2 --device cuda:1 \
--dataset MNIST \
--split_type diri --cncntrtn 1 --test_size 0.2 --spc 50 \
--model_name ResNet10 --hidden_size 64 --resize 32 \
--algorithm fedevg --eval_fraction 1 --eval_type both --eval_every 10 --eval_metrics acc1 fid \
--R 100 --K 10 --C 1. --E 1 --B 64 --server_lr 10. \
--optimizer Adam --lr 0.0001 --weight_decay 1e-4 --lr_decay 0.999 --lr_decay_step 10 --criterion CrossEntropyLoss &
sleep 1

python3 main.py \
--exp_name "MNIST_FedEvg_2 (K=50; alpha=0.01)" --seed 2 --device cuda:2 \
--dataset MNIST \
--split_type diri --cncntrtn 1 --test_size 0.2 --spc 50 \
--model_name ResNet10 --hidden_size 64 --resize 32 \
--algorithm fedevg --eval_fraction 1 --eval_type both --eval_every 50 --eval_metrics acc1 fid \
--R 500 --K 50 --C 0.2 --E 1 --B 64 --server_lr 10. \
--optimizer Adam --lr 0.0001 --weight_decay 1e-4 --lr_decay 0.999 --lr_decay_step 50 --criterion CrossEntropyLoss 
} &&
sleep 1

{
python3 main.py \
--exp_name "MNIST_FedEvg_2 (K=10; alpha=1)" --seed 2 --device cuda:0 \
--dataset MNIST \
--split_type diri --cncntrtn 1 --test_size 0.2 --spc 50 \
--model_name ResNet10 --hidden_size 64 --resize 32 \
--algorithm fedevg --eval_fraction 1 --eval_type both --eval_every 10 --eval_metrics acc1 fid \
--R 100 --K 10 --C 1. --E 1 --B 64 --server_lr 10. \
--optimizer Adam --lr 0.0001 --weight_decay 1e-4 --lr_decay 0.999 --lr_decay_step 10 --criterion CrossEntropyLoss &
sleep 1

python3 main.py \
--exp_name "MNIST_FedEvg_2 (K=50; alpha=1)" --seed 2 --device cuda:1 \
--dataset MNIST \
--split_type diri --cncntrtn 1 --test_size 0.2 --spc 50 \
--model_name ResNet10 --hidden_size 64 --resize 32 \
--algorithm fedevg --eval_fraction 1 --eval_type both --eval_every 50 --eval_metrics acc1 fid \
--R 500 --K 50 --C 0.2 --E 1 --B 64 --server_lr 10. \
--optimizer Adam --lr 0.0001 --weight_decay 1e-4 --lr_decay 0.999 --lr_decay_step 50 --criterion CrossEntropyLoss &
sleep 1

python3 main.py \
--exp_name "MNIST_FedEvg_3 (K=10; alpha=0.01)" --seed 3 --device cuda:2 \
--dataset MNIST \
--split_type diri --cncntrtn 1 --test_size 0.2 --spc 50 \
--model_name ResNet10 --hidden_size 64 --resize 32 \
--algorithm fedevg --eval_fraction 1 --eval_type both --eval_every 10 --eval_metrics acc1 fid \
--R 100 --K 10 --C 1. --E 1 --B 64 --server_lr 10. \
--optimizer Adam --lr 0.0001 --weight_decay 1e-4 --lr_decay 0.999 --lr_decay_step 10 --criterion CrossEntropyLoss &
sleep 1

python3 main.py \
--exp_name "MNIST_FedEvg_3 (K=50; alpha=0.01)" --seed 3 --device cuda:0 \
--dataset MNIST \
--split_type diri --cncntrtn 1 --test_size 0.2 --spc 50 \
--model_name ResNet10 --hidden_size 64 --resize 32 \
--algorithm fedevg --eval_fraction 1 --eval_type both --eval_every 50 --eval_metrics acc1 fid \
--R 500 --K 50 --C 0.2 --E 1 --B 64 --server_lr 10. \
--optimizer Adam --lr 0.0001 --weight_decay 1e-4 --lr_decay 0.999 --lr_decay_step 50 --criterion CrossEntropyLoss &
sleep 1

python3 main.py \
--exp_name "MNIST_FedEvg_3 (K=10; alpha=1)" --seed 3 --device cuda:1 \
--dataset MNIST \
--split_type diri --cncntrtn 1 --test_size 0.2 --spc 50 \
--model_name ResNet10 --hidden_size 64 --resize 32 \
--algorithm fedevg --eval_fraction 1 --eval_type both --eval_every 10 --eval_metrics acc1 fid \
--R 100 --K 10 --C 1. --E 1 --B 64 --server_lr 10. \
--optimizer Adam --lr 0.0001 --weight_decay 1e-4 --lr_decay 0.999 --lr_decay_step 10 --criterion CrossEntropyLoss &
sleep 1

python3 main.py \
--exp_name "MNIST_FedEvg_3 (K=50; alpha=1)" --seed 3 --device cuda:2 \
--dataset MNIST \
--split_type diri --cncntrtn 1 --test_size 0.2 --spc 50 \
--model_name ResNet10 --hidden_size 64 --resize 32 \
--algorithm fedevg --eval_fraction 1 --eval_type both --eval_every 50 --eval_metrics acc1 fid \
--R 500 --K 50 --C 0.2 --E 1 --B 64 --server_lr 10. \
--optimizer Adam --lr 0.0001 --weight_decay 1e-4 --lr_decay 0.999 --lr_decay_step 50 --criterion CrossEntropyLoss 
} &&
sleep 1

echo "...done!"
done