NAME=mepnet_train

DATA_ROOT=data/datasets
DATA_TRAIN=$DATA_ROOT/synthetic_train
DATA_VAL=$DATA_ROOT/synthetic_val

PORT=8019

OMP_NUM_THREADS=1 \
#python train_kp.py \
python -um torch.distributed.launch --nproc_per_node=4 --master_port=12402 train_kp.py \
    --use_ddp \
    --dataroot $DATA_TRAIN  \
    --display_id -1 \
    --no_html \
    --name $NAME --model hourglass_shape_cond \
    --dataset_mode legokps_shape_cond --display_port $PORT \
    --seed -1 \
    --niter 15 \
    --train_brick_pose \
    --voxel_brick \
    --occs_h5 \
    --occ_out_channels 8 \
    --reg_offset \
    --lbd_h 1 \
    --lbd_r 0.1 \
    --lbd_t 1 \
    --max_val_dataset_size 320 \
    --lr 0.00025 \
    --vis_freq 19200 \
    --save_epoch_freq 1 \
    --save_latest_freq 19200 \
    --gpu_ids 0 \
    --lr_policy plateau \
    --lr_decay_iters 3200 \
    --lr_decay_rate 0.9 \
    --print_freq 1 \
    --val_dataroot $DATA_VAL \
    --max_dataset_size 10000000 \
    --occ_fmap_size 256 \
    --crop_brick_occs \
    --batch_size 16 \
    --acc_grad 1 \
    --max_objs 10 \
    --max_brick_types 5 \
    --num_bricks_single_forward 5 \
    --num_threads 2 \
    --lr_patience 1 \
    --kp_sup \
    --camera_jitter 0 \
    --cbrick_brick_kp \
    --num_stacks 2 \
    --img_type laplacian \
    --brick_emb_dim 64 \
    --load_mask \
    --predict_masks \
    --assoc_emb_n_samples 1000 \
    --symmetry_aware_rotation_label \
    --no_coordconv \

echo "Done"
