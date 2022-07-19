NAME=direct3d

CKPT_DIR=checkpoints/

EPOCH=latest

DATASETS=('data/eval_datasets/synthetic_test/' 'data/eval_datasets/classics' 'data/eval_datasets/architecture')
ALIASES=('synthetic_eval_autoreg' 'classics_eval_autoreg' 'architecture_eval_autoreg')
NS=('20' '20' '20')

for i in 0 1 2
do
  DATA_TEST=${DATASETS[i]}
  DATASET_ALIAS=${ALIASES[i]}
  N_SETS=${NS[i]}
  echo $DATA_TEST
  echo $DATASET_ALIAS
  echo $N_SETS
  JAC_QUIET=yes CUDA_LAUNCH_BLOCKING=0 python eval.py \
      --checkpoints_dir $CKPT_DIR \
      --dataroot $DATA_TEST \
      --name $NAME --model hourglass_trans \
      --dataset_mode legokps_shape_cond  \
      --batch_size 5 \
      --epoch $EPOCH \
      --num_threads 0 \
      --occ_out_channels 8 \
      --occ_fmap_size 256 \
      --load_bbox \
      --num_threads 0 \
      --serial_batches \
      --dataset_alias  $DATASET_ALIAS\
      --camera_jitter 0 \
      --max_objs 50 \
      --max_brick_types 5 \
      --n_vis 100 \
      --crop_brick_occs \
      --num_bricks_single_forward 5 \
      --kp_sup \
      --num_stacks 2 \
      --cbrick_brick_kp \
      --img_type laplacian \
      --reg_offset \
      --max_dataset_size 300 \
      --load_lpub \
      --top_center \
      --symmetry_aware_rotation_label \
      --max_dataset_size 300 \
      --n_set $N_SETS \
      --load_bricks \
      --autoregressive_inference \
      --allow_invisible \
      --output_pred_json \


done
echo "Done"
