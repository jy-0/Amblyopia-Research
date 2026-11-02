"""Controlled original-image comparison: released versus locally reproduced RFG."""
import argparse
import copy
import csv
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
import sklearn
import torch
from torch.utils.data import DataLoader

from experiment_anisometropia_frozen import (
    AuditedPairs, evaluate, load_encoder, pair_features, save_json, sha256,
)
from experiment_raw_amblyopia import load_baseline, relabel_records
from experiment_raw_direct_concat import fit_classifier
from report_raw_three_splits import load_and_validate, RUNS
from train_reflection_image import seed_everything

ROOT = Path(__file__).resolve().parent / 'outputs'
SEEDS = (42, 46, 43)
TASKS = ('anisometropia', 'amblyopia')
SPLITS = ('train', 'val', 'test')
METRICS = ('auc', 'acc', 'balanced_acc', 'precision', 'recall', 'f1')
PROTOCOL_KEYS = (
    'encoder', 'image_kind', 'normalization', 'normalization_mean', 'normalization_std',
    'image_shape', 'eye_feature_dim', 'pair_feature_dim', 'pair_fusion', 'output_shape',
    'encoder_parameters', 'trainable_parameters', 'head', 'scaler_fit_split', 'C_selection',
    'threshold_selection', 'augmentation', 'mask_input', 'numerical_features',
)


def record_identity(records):
    return [{k: record[k] for k in ('record_id', 'patient_key', 'eyes')} for record in records]


def check_metrics(records, probabilities, result):
    for key in ('metrics_at_0p5', 'metrics_at_val_threshold'):
        threshold = .5 if key == 'metrics_at_0p5' else result['validation_threshold']
        for split in SPLITS:
            ix = np.array([r['split'] == split for r in records])
            labels = np.array([r['label'] for r in records])[ix]
            measured = evaluate(labels, np.asarray(probabilities)[ix], threshold)
            if measured['confusion_matrix'] != result[key][split]['confusion_matrix']:
                raise ValueError('Confusion matrix does not match predictions')
            for metric, value in measured.items():
                if metric != 'confusion_matrix':
                    np.testing.assert_allclose(value, result[key][split][metric], rtol=0, atol=1e-12)


def fit_and_verify_old(records, eyes, result, predictions):
    for record, prediction in zip(records, predictions):
        if any(record[k] != prediction[k] for k in ('record_id', 'patient_key', 'split', 'label')):
            raise ValueError('Baseline patient, visit, split or label changed')
    if len(records) != len(predictions):
        raise ValueError('Baseline prediction count changed')
    labels = np.array([r['label'] for r in records])
    split = np.array([r['split'] for r in records])
    fit = fit_classifier(pair_features(eyes).astype(np.float64), labels, split, result['seed'])
    if fit['selected_C'] != result['selected_C']:
        raise ValueError('Baseline validation C selection not reproduced')
    np.testing.assert_allclose(fit['validation_threshold'], result['validation_threshold'], rtol=0, atol=1e-12)
    np.testing.assert_allclose(fit['probability'], [p['probability'] for p in predictions], rtol=0, atol=1e-12)
    check_metrics(records, fit['probability'], result)
    return fit


def load_comparisons():
    runs = load_and_validate()
    verified_audit, _, released, _, released_eyes, _ = load_baseline(RUNS[42])
    with Path(verified_audit['source']).open(encoding='utf-8-sig', newline='') as handle:
        source = list(csv.DictReader(handle))
    cases = {}
    for seed in SEEDS:
        run = runs[seed]
        folder = Path(run['folder']) / 'retfound-green_raw_checkpoint'
        if record_identity(run['audit']['records']) != record_identity(verified_audit['records']):
            raise ValueError('Per-eye image ordering or crops differ across seeds')
        with np.load(folder/'features_and_head.npz', allow_pickle=False) as cache:
            np.testing.assert_array_equal(cache['eyes'], released_eyes)
        cases[('anisometropia', seed)] = {
            'records': run['audit']['records'], 'result': run['results'],
            'predictions': run['predictions'], 'folder': folder,
        }
        weak_folder = ROOT/f'amblyopia_raw_frozen_seed{seed}_same544'
        weak_audit = json.loads((weak_folder/'data_audit.json').read_text())
        weak_result = json.loads((weak_folder/'results.json').read_text())
        weak_records = relabel_records(run['audit']['records'], source)
        if weak_audit['records'] != weak_records:
            raise ValueError('Prior amblyopia labels or patient subset differ')
        if record_identity(weak_records) != record_identity(verified_audit['records']):
            raise ValueError('Image input differs across targets')
        with np.load(weak_folder/'features_and_head.npz', allow_pickle=False) as cache:
            np.testing.assert_array_equal(cache['eyes'], released_eyes)
        cases[('amblyopia', seed)] = {
            'records': weak_records, 'result': weak_result,
            'predictions': json.loads((weak_folder/'predictions.json').read_text()), 'folder': weak_folder,
        }
    for (task, seed), case in cases.items():
        if case['result']['seed'] != seed or case['result']['task'] != f'patient-level {task}':
            raise ValueError('Baseline task or seed mismatch')
        for key in PROTOCOL_KEYS:
            if case['result'][key] != released[key]:
                raise ValueError(f'Baseline architecture/protocol differs: {key}')
        if case['result']['weights_sha256'] != released['weights_sha256']:
            raise ValueError('Different released weights in baseline experiments')
        fit_and_verify_old(case['records'], released_eyes, case['result'], case['predictions'])
        print(f'BASELINE VERIFIED {task} seed={seed}', flush=True)
    return verified_audit, released, released_eyes, cases


def encode(records, weights, device, prefix, stop=None):
    seed_everything(42)
    model, size, dim = load_encoder('retfound-green', weights, device)
    assert not model.training and not any(p.requires_grad for p in model.parameters())
    dataset = AuditedPairs(records if stop is None else records[:stop], 'raw', size, 'checkpoint')
    encoded = []
    with torch.inference_mode():
        for images in DataLoader(dataset, batch_size=4, shuffle=False):
            features = model(images.flatten(0, 1).to(device))
            encoded.append(features.cpu().numpy().reshape(len(images), 2, dim))
            done = sum(len(x) for x in encoded)
            if done % 100 == 0 or done == len(dataset):
                print(f'ENCODE {prefix} {done}/{len(dataset)}', flush=True)
    eyes = np.concatenate(encoded)
    assert eyes.shape == (len(dataset), 2, 384) and np.isfinite(eyes).all()
    del model
    torch.cuda.empty_cache()
    return eyes


def aggregate(cases):
    result = {}
    for task in TASKS:
        result[task] = {}
        for model in ('released', 'reproduced'):
            result[task][model] = {}
            for threshold in ('metrics_at_0p5', 'metrics_at_val_threshold'):
                result[task][model][threshold] = {
                    split: {metric: {'mean': float(np.mean(values)), 'sample_std': float(np.std(values, ddof=1))}
                            for metric in METRICS
                            for values in [[cases[task][str(seed)][model][threshold][split][metric] for seed in SEEDS]]}
                    for split in SPLITS}
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=Path, default=ROOT/'retfound_green_reproduction/paper_seed42/retfoundgreen_reproduced_statedict.pth')
    parser.add_argument('--output', type=Path, default=ROOT/'retfound_green_reproduced_raw_comparison_20261020')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    if args.output.exists() and not args.resume:
        raise FileExistsError(f'Refusing to overwrite {args.output}')
    if not args.weights.is_file():
        raise FileNotFoundError(args.weights)
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    device = torch.device(args.device)
    audit, released_result, released_eyes, baselines = load_comparisons()
    args.output.mkdir(parents=True, exist_ok=args.resume)
    weights_hash = sha256(args.weights)
    protocol = {'tasks': list(TASKS), 'seeds': list(SEEDS), 'image_kind': 'raw', 'patients': 544,
                'comparison': 'only encoder checkpoint changes within each task and seed',
                'released_weights': released_result['weights'], 'released_weights_sha256': released_result['weights_sha256'],
                'reproduced_weights': str(args.weights.resolve()), 'reproduced_weights_sha256': weights_hash,
                'image_shape': ['B', 2, 3, 392, 392], 'eye_feature_dim': 384, 'pair_feature_dim': 768,
                'pair_fusion': 'concatenate mean and absolute difference', 'trainable_parameters': 769,
                'C_grid': [.001, .01, .1, 1., 10.], 'C_selection': 'validation AUC; smaller C breaks ties',
                'primary_threshold': .5, 'secondary_threshold': 'validation balanced accuracy only',
                'normalization_mean': [.5]*3, 'normalization_std': [.5]*3, 'augmentation': False,
                'mask_input': False, 'numerical_features': [], 'scaler_fit_split': 'train',
                'device': str(device), 'torch': torch.__version__, 'sklearn': sklearn.__version__,
                'source': audit['source'], 'source_sha256': audit['source_sha256'],
                'baseline_results_refitted': True, 'all_1088_raw_image_hashes_verified': True,
                'started_at': datetime.now(timezone.utc).isoformat()}
    protocol_path = args.output/'protocol.json'
    if args.resume and protocol_path.exists():
        old_protocol = json.loads(protocol_path.read_text())
        for key in ('released_weights_sha256', 'reproduced_weights_sha256', 'source_sha256'):
            if old_protocol[key] != protocol[key]:
                raise ValueError(f'Resume input changed: {key}')
    save_json(protocol_path, protocol)
    save_json(args.output/'image_input_audit.json', {'records': record_identity(audit['records']),
                                                   'source_sha256': audit['source_sha256']})
    # Recompute a released-model image subset to check software/preprocessing
    # parity with the historical frozen cache before comparing checkpoints.
    control = encode(audit['records'], Path(released_result['weights']), device, 'released_control', stop=16)
    difference = float(np.max(np.abs(control-released_eyes[:16])))
    np.testing.assert_allclose(control, released_eyes[:16], rtol=1e-5, atol=1e-5)
    save_json(args.output/'released_encoder_control.json', {'patients': 16, 'images': 32,
              'max_absolute_feature_error': difference, 'rtol': 1e-5, 'atol': 1e-5, 'passed': True})
    cache = args.output/'reproduced_eye_embeddings.npz'
    if args.resume and cache.exists():
        with np.load(cache, allow_pickle=False) as values:
            if values['weights_sha256'].item() != weights_hash:
                raise ValueError('Reproduced embedding cache weight mismatch')
            eyes = values['eyes'].copy()
    else:
        eyes = encode(audit['records'], args.weights, device, 'reproduced')
        np.savez_compressed(cache, eyes=eyes, weights_sha256=weights_hash)
    if eyes.shape != released_eyes.shape or not np.isfinite(eyes).all():
        raise ValueError('Reproduced cache geometry differs')
    x = pair_features(eyes).astype(np.float64)
    np.testing.assert_array_equal(x, pair_features(eyes[:, ::-1]).astype(np.float64))
    experiments = {task: {} for task in TASKS}
    provenance = []
    for task in TASKS:
        for seed in SEEDS:
            baseline = baselines[(task, seed)]
            records = baseline['records']
            labels = np.array([r['label'] for r in records])
            split = np.array([r['split'] for r in records])
            fitted = fit_classifier(x, labels, split, seed)
            folder = args.output/f'{task}_seed{seed}'
            folder.mkdir(exist_ok=args.resume)
            result = {key: copy.deepcopy(baseline['result'][key]) for key in PROTOCOL_KEYS}
            result.update(task=f'patient-level {task}', seed=seed, split_counts=baseline['result']['split_counts'],
                          weights=str(args.weights.resolve()), weights_sha256=weights_hash,
                          weight_source='local 75000-image 120-epoch reproduction',
                          baseline_results=str(baseline['folder']/'results.json'),
                          frozen_feature_cache=str(cache.resolve()), frozen_feature_cache_sha256=sha256(cache),
                          **{key: fitted[key] for key in ('selected_C', 'validation_threshold', 'trials',
                                                       'metrics_at_0p5', 'metrics_at_val_threshold')})
            for key in PROTOCOL_KEYS:
                if result[key] != baseline['result'][key]:
                    raise ValueError(f'Unintended protocol difference: {key}')
            predictions = [{**{key: record[key] for key in ('record_id', 'patient_key', 'split', 'label')},
                            'probability': float(p), 'prediction_0p5': int(p >= .5),
                            'prediction_val_threshold': int(p >= fitted['validation_threshold'])}
                           for record, p in zip(records, fitted['probability'])]
            check_metrics(records, fitted['probability'], result)
            save_json(folder/'data_audit.json', {'records': records, 'split_counts': result['split_counts'],
                                               'only_changed_input': 'encoder checkpoint'})
            save_json(folder/'results.json', result)
            save_json(folder/'predictions.json', predictions)
            np.savez_compressed(folder/'features_and_head.npz', eyes=eyes,
                                scaler_mean=fitted['scaler'].mean_, scaler_scale=fitted['scaler'].scale_,
                                coefficient=fitted['head'].coef_, intercept=fitted['head'].intercept_)
            experiments[task][str(seed)] = {'released': baseline['result'], 'reproduced': result}
            provenance.append({'task': task, 'seed': seed, 'baseline_directory': str(baseline['folder']),
                               'results_sha256': sha256(baseline['folder']/'results.json'),
                               'predictions_sha256': sha256(baseline['folder']/'predictions.json')})
            print(f'RESULT {task} seed={seed} '+json.dumps(result['metrics_at_0p5']['test']), flush=True)
    summary = {'protocol': protocol, 'experiments': experiments, 'aggregate': aggregate(experiments),
               'baseline_provenance': provenance, 'released_control_max_absolute_feature_error': difference,
               'verification': 'All six released heads refitted; exact image hashes/visits/patient splits/labels preserved; all new split metrics recomputed.',
               'interpretation': 'Three overlapping patient partitions, not independent external cohorts. Sample SD is not a confidence interval.'}
    save_json(args.output/'summary.json', summary)
    save_json(args.output/'status.json', {'state': 'completed', 'experiments': 6,
                                        'completed_at': datetime.now(timezone.utc).isoformat()})
    print(f'COMPLETE {args.output}', flush=True)


if __name__ == '__main__':
    main()
