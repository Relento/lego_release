TRAIN_DATA_DIR=/svl/u/rcwang/lego/synthetic_train
VAL_DATA_DIR=/svl/u/rcwang/lego/synthetic_val

echo "processing dataset: "$DATA_DIR

export PYTHONPATH=.
python ./scripts/dump_occs_h5.py -d $TRAIN_DATA_DIR
python ./scripts/dump_occs_h5.py -d $VAL_DATA_DIR
echo "dump occ points done"