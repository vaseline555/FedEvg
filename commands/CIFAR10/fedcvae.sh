#!/bin/sh
# 60000 images * (1 - test_size) // K clients // B batches * E iterations (client-side)
# + (10 * num_classes * int(C * K) generated images // B bathces * E epochs) * 2 (server-side)
## -> 60000 iters
# num params. (for communicaiton per client): 836,353 (CVAE decoder)

{
python3 main.py \
--exp_name "CIFAR10_FedCVAE_1 (K=10; alpha=0.01)" --seed 1 --device cuda:0 \
--dataset CIFAR10 \
--split_type diri --cncntrtn 0.01 --test_size 0.2 \
--model_name CVAE --hidden_size 64 --resize 32 --spc 20 \
--algorithm fedcvae --eval_fraction 1 --eval_type both --eval_every 1 --eval_metrics acc1 fid \
--R 1 --K 10 --C 1 --E 20 --B 64 \
--optimizer Adam --lr 0.001 --weight_decay 1e-4 --criterion MSELoss &
sleep 1

python3 main.py \
--exp_name "CIFAR10_FedCVAE_1 (K=50; alpha=0.01)" --seed 1 --device cuda:1 \
--dataset CIFAR10 \
--split_type diri --cncntrtn 0.01 --test_size 0.2 \
--model_name CVAE --hidden_size 64 --resize 32 --spc 20 \
--algorithm fedcvae --eval_fraction 1 --eval_type both --eval_every 1 --eval_metrics acc1 fid \
--R 1 --K 50 --C 1 --E 100 --B 64 \
--optimizer Adam --lr 0.001 --weight_decay 1e-4 --criterion MSELoss &
sleep 1

python3 main.py \
--exp_name "CIFAR10_FedCVAE_1 (K=10; alpha=1)" --seed 1 --device cuda:2 \
--dataset CIFAR10 \
--split_type diri --cncntrtn 1 --test_size 0.2 \
--model_name CVAE --hidden_size 64 --resize 32 --spc 20 \
--algorithm fedcvae --eval_fraction 1 --eval_type both --eval_every 1 --eval_metrics acc1 fid \
--R 1 --K 10 --C 1 --E 20 --B 64 \
--optimizer Adam --lr 0.001 --weight_decay 1e-4 --criterion MSELoss &
sleep 1

python3 main.py \
--exp_name "CIFAR10_FedCVAE_1 (K=50; alpha=1)" --seed 1 --device cuda:0 \
--dataset CIFAR10 \
--split_type diri --cncntrtn 1 --test_size 0.2 \
--model_name CVAE --hidden_size 64 --resize 32 --spc 20 \
--algorithm fedcvae --eval_fraction 1 --eval_type both --eval_every 1 --eval_metrics acc1 fid \
--R 1 --K 50 --C 1 --E 100 --B 64 \
--optimizer Adam --lr 0.001 --weight_decay 1e-4 --criterion MSELoss &
sleep 1

python3 main.py \
--exp_name "CIFAR10_FedCVAE_2 (K=10; alpha=0.01)" --seed 2 --device cuda:1 \
--dataset CIFAR10 \
--split_type diri --cncntrtn 0.01 --test_size 0.2 \
--model_name CVAE --hidden_size 64 --resize 32 --spc 20 \
--algorithm fedcvae --eval_fraction 1 --eval_type both --eval_every 1 --eval_metrics acc1 fid \
--R 1 --K 10 --C 1 --E 20 --B 64 \
--optimizer Adam --lr 0.001 --weight_decay 1e-4 --criterion MSELoss &
sleep 1

python3 main.py \
--exp_name "CIFAR10_FedCVAE_2 (K=50; alpha=0.01)" --seed 2 --device cuda:2 \
--dataset CIFAR10 \
--split_type diri --cncntrtn 0.01 --test_size 0.2 \
--model_name CVAE --hidden_size 64 --resize 32 --spc 20 \
--algorithm fedcvae --eval_fraction 1 --eval_type both --eval_every 1 --eval_metrics acc1 fid \
--R 1 --K 50 --C 1 --E 100 --B 64 \
--optimizer Adam --lr 0.001 --weight_decay 1e-4 --criterion MSELoss 
} &&
sleep 1

{
python3 main.py \
--exp_name "CIFAR10_FedCVAE_2 (K=10; alpha=1)" --seed 2 --device cuda:0 \
--dataset CIFAR10 \
--split_type diri --cncntrtn 1 --test_size 0.2 \
--model_name CVAE --hidden_size 64 --resize 32 --spc 20 \
--algorithm fedcvae --eval_fraction 1 --eval_type both --eval_every 1 --eval_metrics acc1 fid \
--R 1 --K 10 --C 1 --E 20 --B 64 \
--optimizer Adam --lr 0.001 --weight_decay 1e-4 --criterion MSELoss &
sleep 1

python3 main.py \
--exp_name "CIFAR10_FedCVAE_2 (K=50; alpha=1)" --seed 2 --device cuda:1 \
--dataset CIFAR10 \
--split_type diri --cncntrtn 1 --test_size 0.2 \
--model_name CVAE --hidden_size 64 --resize 32 --spc 20 \
--algorithm fedcvae --eval_fraction 1 --eval_type both --eval_every 1 --eval_metrics acc1 fid \
--R 1 --K 50 --C 1 --E 100 --B 64 \
--optimizer Adam --lr 0.001 --weight_decay 1e-4 --criterion MSELoss &
sleep 1

python3 main.py \
--exp_name "CIFAR10_FedCVAE_3 (K=10; alpha=0.01)" --seed 3 --device cuda:2 \
--dataset CIFAR10 \
--split_type diri --cncntrtn 0.01 --test_size 0.2 \
--model_name CVAE --hidden_size 64 --resize 32 --spc 20 \
--algorithm fedcvae --eval_fraction 1 --eval_type both --eval_every 1 --eval_metrics acc1 fid \
--R 1 --K 10 --C 1 --E 20 --B 64 \
--optimizer Adam --lr 0.001 --weight_decay 1e-4 --criterion MSELoss &
sleep 1

python3 main.py \
--exp_name "CIFAR10_FedCVAE_3 (K=50; alpha=0.01)" --seed 3 --device cuda:0 \
--dataset CIFAR10 \
--split_type diri --cncntrtn 0.01 --test_size 0.2 \
--model_name CVAE --hidden_size 64 --resize 32 --spc 20 \
--algorithm fedcvae --eval_fraction 1 --eval_type both --eval_every 1 --eval_metrics acc1 fid \
--R 1 --K 50 --C 1 --E 100 --B 64 \
--optimizer Adam --lr 0.001 --weight_decay 1e-4 --criterion MSELoss &
sleep 1

python3 main.py \
--exp_name "CIFAR10_FedCVAE_3 (K=10; alpha=1)" --seed 3 --device cuda:1 \
--dataset CIFAR10 \
--split_type diri --cncntrtn 1 --test_size 0.2 \
--model_name CVAE --hidden_size 64 --resize 32 --spc 20 \
--algorithm fedcvae --eval_fraction 1 --eval_type both --eval_every 1 --eval_metrics acc1 fid \
--R 1 --K 10 --C 1 --E 20 --B 64 \
--optimizer Adam --lr 0.001 --weight_decay 1e-4 --criterion MSELoss &
sleep 1

python3 main.py \
--exp_name "CIFAR10_FedCVAE_3 (K=50; alpha=1)" --seed 3 --device cuda:2 \
--dataset CIFAR10 \
--split_type diri --cncntrtn 1 --test_size 0.2 \
--model_name CVAE --hidden_size 64 --resize 32 --spc 20 \
--algorithm fedcvae --eval_fraction 1 --eval_type both --eval_every 1 --eval_metrics acc1 fid \
--R 1 --K 50 --C 1 --E 100 --B 64 \
--optimizer Adam --lr 0.001 --weight_decay 1e-4 --criterion MSELoss
} &&
sleep 1

echo "...done!"
done