NAME=mepnet

CKPT_DIR=checkpoints/

DATASETS=('data/eval_datasets/synthetic_test/' 'data/eval_datasets/classics' 'data/eval_datasets/architecture')
ALIASES=('synthetic_eval_autoreg' 'classics_eval_autoreg' 'architecture_eval_autoreg')
NS=('20' '20' '20')

EPOCH=latest
for i in 0 1 2
do
  DATA_TEST=${DATASETS[i]}
  DATASET_ALIAS=${ALIASES[i]}
  N_SETS=${NS[i]}
  echo Evaluating $DATA_TEST
  echo Experiment alias: $DATASET_ALIAS
  echo Number of sets: $N_SETS "(including the submodules)"
  JAC_QUIET=yes CUDA_LAUNCH_BLOCKING=0 python eval.py \
      --checkpoints_dir $CKPT_DIR \
      --dataroot $DATA_TEST \
      --name $NAME --model hourglass_shape_cond \
      --dataset_mode legokps_shape_cond  \
      --batch_size 10 \
      --epoch $EPOCH \
      --num_threads 0 \
      --occ_out_channels 8 \
      --occ_fmap_size 256 \
      --load_bbox \
      --load_conn \
      --top_center \
      --num_threads 0 \
      --load_bricks \
      --serial_batches \
      --dataset_alias  $DATASET_ALIAS\
      --camera_jitter 0 \
      --max_objs 50 \
      --max_brick_types 5 \
      --n_vis 200 \
      --crop_brick_occs \
      --num_bricks_single_forward 5 \
      --kp_sup \
      --num_stacks 2 \
      --cbrick_brick_kp \
      --img_type laplacian \
      --load_mask \
      --max_dataset_size 300 \
      --load_mask \
      --n_set $N_SETS \
      --search_ref_mask \
      --allow_invisible \
      --oracle_percentage 0.0 \
      --symmetry_aware_rotation_label \
      --no_coordconv \
      --load_lpub \
      --autoregressive_inference \
      --output_pred_json \
      --predict_masks \
      # --render_pred_json
      # Uncomment the above line to render visualization of predicted bricks,
      # which will slow down the evaluation because of rendering.



done
echo "Done"
