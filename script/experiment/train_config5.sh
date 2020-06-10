#!/usr/bin/env python

python ./script/experiment/train_deepmar_resnet50.py \
    --sys_device_ids="(3,)" \
    --dataset=rap \
    --partition_idx=0 \
    --split=trainval \
    --test_split=test \
    --batch_size=32 \
    --resize="(224,224)" \
    --exp_subpath=deepmar_resnet50 \
    --new_params_lr=0.001 \
    --finetuned_params_lr=0.001 \
    --staircase_decay_at_epochs="(50,100)" \
    --total_epochs=150 \
    --epochs_per_val=10\
    --epochs_per_save=50 \
    --drop_pool5=True \
    --drop_pool5_rate=0.5 \
    --run=1 \
    --resume=False \
    --ckpt_file= \
    --test_only=False \
    --coeff_threshold=0.3 \
    --p=0.2 \
    --model_weight_file= \
