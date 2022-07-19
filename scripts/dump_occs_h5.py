import argparse
import os
import pickle

import h5py
import numpy as np
from tqdm import tqdm


def main():
    dataset_dir = os.path.abspath(args.data_dir)
    assert os.path.exists(dataset_dir), dataset_dir
    occ_h5_file = 'occs.h5'
    id_f_file = 'occ_idxs.pkl'

    occ_dirs = []

    dump_dir = os.path.join(dataset_dir, 'occs')
    os.makedirs(dump_dir, exist_ok=True)

    os.chdir(dump_dir)
    print('Moving to ', dump_dir)

    for l in os.listdir(dataset_dir):
        if l not in ['occs', 'metadata']:
            occ_dirs.append(os.path.join(dataset_dir, l, 'occs.pkl'))

    occ_idxs = {}

    cur = 0
    hf = h5py.File(occ_h5_file, 'w')
    id_f = open(id_f_file, 'wb')

    dset = hf.create_dataset('dataset', (10 ** 5, 3), maxshape=(None, 3), dtype=np.uint8)
    for occ_dir in tqdm(occ_dirs):
        with open(occ_dir, 'rb') as f:
            d = pickle.load(f)
        for i, occ in enumerate(d):
            dset.resize(cur + occ.shape[0], axis=0)
            dset[cur:cur + occ.shape[0], :] = occ
            occ_idxs[(occ_dir, i)] = (cur, cur + occ.shape[0])
            cur += occ.shape[0]

    pickle.dump(occ_idxs, id_f)
    hf.close()
    id_f.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data_dir', required=True, type=str,
                        help='dataset to be processed, e.g. data/datasets/my_data/train/')
    # assume data_dir contains folders 000000, 000001, ...

    args = parser.parse_args()

    main()
