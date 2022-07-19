import os

models = [
    'mepnet',
    'direct3d',
]

step_datasets = [
    'synthetic_eval',
    'classics_eval',
    'architecture_eval',
]

autoreg_datasets = [
    'synthetic_eval_autoreg',
    'classics_eval_autoreg',
    'architecture_eval_autoreg',
]


def get_output_dir(model, dataset):
    return f'results/{model}/{dataset}/output'


for m in models:
    print('Evaluating model:', m)
    for d in step_datasets:
        print('Dataset:', d, ' Mode: Per Step')
        os.system(f'PYTHONPATH=. python chamfer_distance_eval.py --json_path {get_output_dir(m, d)}')
    for d in autoreg_datasets:
        print('Dataset:', d, ' Mode: From Scratch')
        os.system(f'PYTHONPATH=. python chamfer_distance_eval.py --json_path {get_output_dir(m, d)} --compute_final')
    print()
