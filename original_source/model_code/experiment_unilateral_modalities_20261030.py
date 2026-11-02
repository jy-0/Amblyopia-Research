"""Patient-matched image/table/fusion probes for unilateral amblyopia."""

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, average_precision_score, balanced_accuracy_score,
    confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score, roc_curve,
)
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from threadpoolctl import threadpool_limits

from audit_astigmatism_workbook import age, num, text


ROOT = Path(__file__).resolve().parent / 'outputs'
SUBSET = ROOT / 'splits/normal_unilateral_strict_special_20261030'
PARENT = ROOT / 'splits/analysis_1024_label_only_20261028'
CACHE = ROOT / 'expanded1524_raw_three_seed_20261028/frozen_eye_embeddings.npz'
DEFAULT_OUT = ROOT / 'unilateral_modalities_three_seed_20261030'
SEEDS = (42, 46, 43)
SPLITS = ('train', 'val', 'test')
C_GRID = (.001, .01, .1, 1., 10.)
GAMMA_GRID = ('scale', .0001, .001, .01)
FEATURES = ('性别', '年龄', '右球镜', '右柱镜', '右轴位', '左球镜', '左柱镜', '左轴位')
PRIMARY_MODES = ('image', 'table8', 'fusion')
POLICIES = ('default', 'validation_bacc', 'validation_sens90')


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def clinical_records(records):
    new = {r['excel_row']: r for r in read(ROOT / 'audits/dataset_1024_20261028_canonical/workbook_records.json')}
    with Path('/root/the_dataset 1.csv').open(encoding='utf-8-sig', newline='') as handle:
        source = list(csv.DictReader(handle))
    old = {r['record_id']: r for r in source}
    assert len(old) == len(source)
    raw, details = [], []
    for record in records:
        current = record.get('source_excel_row') is not None
        row = new[record['source_excel_row']] if current else old[record['record_id']]
        assert (row['normalized_name'] if current else row['patient_key']) == record['patient_key']
        if current:
            assert row['weak_label'] == record['weak_label'] and row['aniso_label'] == record['aniso_label']
        sex = text(row.get('性别'))
        if sex not in ('男', '女'):
            raise ValueError(f'Unknown sex: {sex!r}')
        a = age(row.get('年龄/检查时年龄（周岁）' if current else '年龄'))
        values = [float(sex == '女'), a]
        for side in ('右', '左'):
            for field in ('球镜', '柱镜', '轴位'):
                key = f'{side}{field}' + ('（原始文本）' if current else '')
                value = row.get(key)
                parsed = num(value)
                if text(value) and parsed is None:
                    raise ValueError(f'Unparsed refraction: {record["name"]} {key} {value!r}')
                if field == '轴位' and parsed is not None and not 0 <= parsed <= 180:
                    raise ValueError(f'Axis out of range: {parsed}')
                values.append(parsed)
        numbers = np.array([np.nan if v is None else v for v in values], dtype=np.float64)
        raw.append(numbers)
        info = {k: record.get(k) for k in ('patient_key', 'name', 'record_id', 'weak_label', 'aniso_label', 'weak_eye', 'image_origin')}
        info.update(sex=sex, age=a, refraction_type=row.get('验光类型'),
                    diagnosis_reference=row.get('门诊诊断-参考用'),
                    raw_weak_label=row.get('原表弱视判断' if current else 'label_idx'),
                    raw_special_note=row.get('原表特殊情况' if current else '特殊情况'),
                    raw_deprivation=row.get('原表形觉剥夺因素' if current else '形觉剥夺因素'),
                    missing_fields=[FEATURES[i] for i in np.flatnonzero(~np.isfinite(numbers))])
        for i, name in enumerate(FEATURES[2:], 2):
            info[name] = None if not np.isfinite(numbers[i]) else float(numbers[i])
        for side in ('右', '左'):
            info[f'{side}BCVA'] = row.get(f'{side}BCVA' + ('（原始文本）' if current else ''))
            info[f'{side}SE'] = num(row.get(f'{side}SE（D）' if current else f'{side}等效球镜（D）'))
        info['SE差'] = num(row.get('双眼SE差绝对值（D）' if current else '等效球镜差（D）'))
        details.append(info)
    return np.asarray(raw), details


def impute_training_means(raw, split):
    raw, split = np.asarray(raw, dtype=float), np.asarray(split)
    if raw.ndim != 2 or raw.shape[1] != 8 or np.isinf(raw).any() or not (split == 'train').any():
        raise ValueError('Expected eight clinical inputs and a training split')
    means = np.zeros(8, dtype=float)
    # Pool homologous eye fields using observed training values only.
    for columns in ([0], [1], [2, 5], [3, 6], [4, 7]):
        observed = raw[split == 'train'][:, columns].reshape(-1)
        observed = observed[np.isfinite(observed)]
        means[columns] = observed.mean() if len(observed) else 0.
    return np.where(np.isfinite(raw), raw, means), means


def features(image, table, mode):
    if mode == 'image':
        return image, np.ones(image.shape[1])
    if mode == 'table8':
        return table, np.ones(8)
    if mode == 'demo':
        return table[:, :2], np.ones(2)
    if mode == 'image_demo':
        blocks = [image, table[:, :2]]
    elif mode == 'fusion':
        blocks = [image, table]
    else:
        raise ValueError(mode)
    width = sum(x.shape[1] for x in blocks)
    # Equal expected block energy after train-only StandardScaler; fixed before fitting.
    weights = np.concatenate([np.full(x.shape[1], np.sqrt(width / (2 * x.shape[1]))) for x in blocks])
    return np.concatenate(blocks, axis=1), weights


def metrics(y, scores, threshold):
    y, scores = np.asarray(y, dtype=int), np.asarray(scores, dtype=float)
    predicted = scores >= threshold
    (tn, fp), (fn, tp) = confusion_matrix(y, predicted, labels=[0, 1])
    both = len(np.unique(y)) == 2
    return {
        'patients': len(y), 'positive': int(y.sum()), 'negative': int(len(y) - y.sum()),
        'threshold': float(threshold), 'auc': float(roc_auc_score(y, scores)) if both else None,
        'acc': float(accuracy_score(y, predicted)),
        'sensitivity': float(tp / (tp + fn)) if tp + fn else None,
        'specificity': float(tn / (tn + fp)) if tn + fp else None,
        'f1': float(f1_score(y, predicted, zero_division=0)),
        'precision': float(precision_score(y, predicted, zero_division=0)),
        'npv': float(tn / (tn + fn)) if tn + fn else None,
        'balanced_acc': float(balanced_accuracy_score(y, predicted)) if both else None,
        'average_precision': float(average_precision_score(y, scores)) if both else None,
        'referral_rate': float(predicted.mean()),
        'confusion_matrix': [[int(tn), int(fp)], [int(fn), int(tp)]],
    }


def thresholds(y, scores, default):
    fpr, tpr, cuts = roc_curve(y, scores, drop_intermediate=False)
    valid = np.flatnonzero(np.isfinite(cuts))
    best = max(valid, key=lambda i: ((tpr[i] + 1 - fpr[i]) / 2, -abs(cuts[i] - default), -cuts[i]))
    candidates = [i for i in valid if tpr[i] >= .9 - 1e-12]
    high_sensitivity = max(candidates, key=lambda i: (1 - fpr[i], cuts[i]))
    return {'default': float(default), 'validation_bacc': float(cuts[best]), 'validation_sens90': float(cuts[high_sensitivity])}


def fit_probe(x, y, split, weights, classifier, seed):
    train, val = split == 'train', split == 'val'
    scaler = StandardScaler().fit(x[train])
    z_train = scaler.transform(x[train]) * weights
    z_val = scaler.transform(x[val]) * weights
    best, best_auc, selected, trials = None, -np.inf, None, []
    for c in C_GRID:
        for gamma in (GAMMA_GRID if classifier == 'rbf_svm' else (None,)):
            if classifier == 'logistic':
                model = LogisticRegression(C=c, class_weight='balanced', solver='lbfgs',
                                           max_iter=3000, tol=1e-8, random_state=seed)
            else:
                model = SVC(C=c, gamma=gamma, kernel='rbf', class_weight='balanced', probability=False,
                            tol=1e-6, max_iter=100000, cache_size=512, random_state=seed)
            model.fit(z_train, y[train])
            if (classifier == 'logistic' and model.n_iter_.max() >= 3000) or (classifier == 'rbf_svm' and model.fit_status_):
                raise RuntimeError(f'Classifier failed to converge: {classifier} C={c} gamma={gamma}')
            assert model.classes_.tolist() == [0, 1]
            score = model.predict_proba(z_val)[:, 1] if classifier == 'logistic' else model.decision_function(z_val)
            auc = float(roc_auc_score(y[val], score))
            params = {'C': c, **({'gamma': gamma, 'kernel': 'rbf'} if classifier == 'rbf_svm' else {})}
            trials.append({'parameters': params, 'validation_auc': auc})
            if auc > best_auc:
                best, best_auc, selected = model, auc, params
    score_val = best.predict_proba(z_val)[:, 1] if classifier == 'logistic' else best.decision_function(z_val)
    selected_thresholds = thresholds(y[val], score_val, .5 if classifier == 'logistic' else 0.)
    z = scaler.transform(x) * weights
    scores = best.predict_proba(z)[:, 1] if classifier == 'logistic' else best.decision_function(z)
    return scaler, best, scores, selected_thresholds, selected, trials


def identity(r):
    return (r['patient_key'], tuple((e['raw_sha256'], tuple(e['crop_box'])) for e in r['eyes']))


def load_data():
    parent = read(PARENT / 'patient_manifest.json')['records']
    cache_audit = read(ROOT / 'expanded1524_raw_three_seed_20261028/amblyopia_seed42/data_audit.json')['records']
    with np.load(CACHE, allow_pickle=False) as stored:
        eyes = stored['eyes'].copy()
        names = stored['patient_key'].tolist()
        weights_sha = stored['weights_sha256'].item()
    assert [r['patient_key'] for r in parent] == names
    assert [identity(r) for r in parent] == [identity(r) for r in cache_audit]
    assert eyes.shape == (len(parent), 2, 384) and np.isfinite(eyes).all()
    assert sha(ROOT / 'pretrained/retfoundgreen_statedict.pth') == weights_sha
    image = np.concatenate([eyes.mean(1), np.abs(eyes[:, 0] - eyes[:, 1])], axis=1).astype(float)
    raw, clinical = clinical_records(parent)
    for info, eye in zip(clinical, eyes):
        info['图像特征绝对差均值'] = float(np.abs(eye[0] - eye[1]).mean())
        info['双眼图像特征余弦相似度'] = float(eye[0] @ eye[1] / max(np.linalg.norm(eye[0]) * np.linalg.norm(eye[1]), 1e-12))
    excluded = read(SUBSET / 'exclusions.json')['records']
    special = {r['patient_key'] for r in excluded if 'special_condition_nonempty' in r['reasons'] or 'form_deprivation_positive' in r['reasons']}
    unilateral = {r['patient_key'] for r in read(SUBSET / 'patient_manifest.json')['records']}
    strict_all = {r['patient_key'] for r in parent} - special
    assert len(unilateral) == 1100 and len(strict_all) == 1298
    assert unilateral <= strict_all
    for r in parent:
        if r['patient_key'] in strict_all - unilateral:
            assert r['weak_label'] == 1 and r['weak_eye'] == '双眼'
    return parent, image, raw, clinical, {'unilateral': unilateral, 'strict_all_types': strict_all}, weights_sha


def run(args):
    output = args.output
    if output.exists() and not args.resume:
        raise FileExistsError(output)
    output.mkdir(parents=True, exist_ok=True)
    parent, images, raw, clinical, cohorts, weights_sha = load_data()
    protocol = {
        'created_at': datetime.now(timezone.utc).isoformat(), 'task': 'patient-level binary amblyopia',
        'seeds': list(SEEDS), 'primary_runs': 18, 'supplementary_runs': 18,
        'classifiers': ['logistic', 'rbf_svm'],
        'primary_inputs': {'image': 'bilateral raw images only; no demographic input', 'table8': list(FEATURES), 'fusion': 'image embedding concatenated with eight clinical fields'},
        'input_dimensions': {'image': 768, 'table8': 8, 'fusion': 776, 'demo': 2, 'image_demo': 770},
        'encoder': 'official frozen full12 RETFound-Green; shared R/L; 392x392 RGB; 384 features/eye',
        'image_fusion': 'bilateral mean384 + absolute difference384', 'encoder_sha256': weights_sha,
        'feature_cache': str(CACHE), 'cache_sha256': sha(CACHE),
        'table_order': list(FEATURES), 'sex_encoding': {'男': 0, '女': 1},
        'imputation': 'observed training means only; pooled R/L per sphere/cylinder/axis; equivalent missing->zero after observed-statistic normalization; no mask and no feature-based patient exclusions',
        'axis_encoding': 'original numeric axis in degrees; no cyclic expansion or engineered refraction features',
        'modality_balancing': 'after training StandardScaler, each fusion block multiplied by sqrt(total_dimensions/(2*block_dimensions)); equal nominal block energy; fixed, not test-tuned',
        'patient_splits': 'preserve all existing seed-specific patient memberships; never move a historical train patient to test',
        'special_condition': 'exclude any nonempty note across available visits, including general quality notes',
        'form_deprivation': 'exclude provided explicit positive labels',
        'bilateral_control': 'strict_all_types adds only bilateral-amblyopia patients under otherwise identical exclusions',
        'C_grid': list(C_GRID), 'gamma_grid': list(GAMMA_GRID), 'class_weight': 'balanced',
        'parameter_selection': 'validation AUROC only; first candidate breaks ties; no refit on validation',
        'thresholds': 'primary default .5 for LR, 0 for SVM; supplementary validation maximum BACC and maximum specificity subject to validation sensitivity>=90%',
        'augmentation': False, 'mask_input': False, 'test_used_for_selection': False,
        'SVM_score': 'uncalibrated decision margin, not probability', 'script_sha256': sha(__file__),
        'limitations': ['three overlapping historical splits, not independent external cohorts', 'test sets already used in prior exploratory work', 'different group prevalence prevents attributing full-test changes solely to bilateral exclusion', 'missing cylinder/axis is not verified absence of astigmatism', 'direct clinical LR is linear in eight original fields; no engineered absolute refraction differences'],
    }
    if args.resume:
        previous = read(output / 'protocol.json')
        for key in ('script_sha256', 'cache_sha256', 'encoder_sha256'):
            assert previous[key] == protocol[key], key
    else:
        save(output / 'protocol.json', protocol)
    save(output / 'clinical_records.json', clinical)
    np.savez_compressed(output / 'clinical_raw8.npz', raw=raw, patient_key=np.asarray([r['patient_key'] for r in parent]))
    runs = []
    for cohort, names in cohorts.items():
        indices = np.array([i for i, r in enumerate(parent) if r['patient_key'] in names])
        selected = [parent[i] for i in indices]
        folder = output / cohort
        folder.mkdir(exist_ok=True)
        save(folder / 'patient_manifest.json', {'records': selected})
        for seed in SEEDS:
            source = read(PARENT / f'seed{seed}/patient_manifest.json')['records']
            by_name = {r['patient_key']: r for r in source}
            records = [{**r, 'split': by_name[r['patient_key']]['split'], 'label': r['weak_label']} for r in selected]
            y, split = np.array([r['label'] for r in records]), np.array([r['split'] for r in records])
            assert all(set(y[split == s]) == {0, 1} for s in SPLITS)
            seed_dir = folder / f'seed{seed}'
            seed_dir.mkdir(exist_ok=True)
            filled, means = impute_training_means(raw[indices], split)
            audit = {
                'records': records, 'clinical_features': list(FEATURES), 'training_imputation_means': means.tolist(),
                'missing_by_split': {s: dict(zip(FEATURES, (~np.isfinite(raw[indices][split == s])).sum(0).tolist())) for s in SPLITS},
                'patients_with_missing': {s: int((~np.isfinite(raw[indices][split == s])).any(1).sum()) for s in SPLITS},
                'split_counts': {s: {'patients': int((split == s).sum()), 'positive': int(y[split == s].sum()), 'negative': int((1-y[split == s]).sum())} for s in SPLITS},
            }
            save(seed_dir / 'data_audit.json', audit)
            modes = (*PRIMARY_MODES, 'demo', 'image_demo') if cohort == 'unilateral' else ('image',)
            for mode in modes:
                x, block_weights = features(images[indices], filled, mode)
                for classifier in ('logistic', 'rbf_svm'):
                    dest = seed_dir / mode / classifier
                    dest.mkdir(parents=True, exist_ok=True)
                    run_id = f'{cohort}_seed{seed}_{mode}_{classifier}'
                    save(output / 'status.json', {'state': 'fitting', 'current': run_id, 'finished': len(runs)})
                    if args.resume and (dest / 'results.json').exists():
                        runs.append({'run_id': run_id, 'folder': str(dest), **read(dest / 'results.json')})
                        continue
                    print('FIT', run_id, 'n=', len(y), 'dim=', x.shape[1], flush=True)
                    scaler, head, scores, cuts, params, trials = fit_probe(x, y, split, block_weights, classifier, seed)
                    result = {
                        'cohort': cohort, 'seed': seed, 'mode': mode, 'classifier': classifier,
                        'primary': cohort == 'unilateral' and mode in PRIMARY_MODES,
                        'input_dim': x.shape[1], 'patients': len(y), 'thresholds': cuts,
                        'selected_parameters': params, 'trials': trials,
                        'score_type': 'probability' if classifier == 'logistic' else 'uncalibrated_margin',
                        'metrics': {policy: {s: metrics(y[split == s], scores[split == s], cuts[policy]) for s in SPLITS} for policy in POLICIES},
                    }
                    artifact = {'classifier': head, 'scaler': scaler, 'block_weights': block_weights,
                                'clinical_imputation_means': means, 'table_order': FEATURES,
                                'mode': mode, 'cohort': cohort, 'seed': seed, 'thresholds': cuts,
                                'encoder_sha256': weights_sha}
                    joblib.dump(artifact, dest / 'classifier.joblib')
                    restored = joblib.load(dest / 'classifier.joblib')
                    z = restored['scaler'].transform(x) * restored['block_weights']
                    checked = restored['classifier'].predict_proba(z)[:, 1] if classifier == 'logistic' else restored['classifier'].decision_function(z)
                    np.testing.assert_allclose(scores, checked, rtol=0, atol=1e-12)
                    predictions = [{
                        'patient_key': r['patient_key'], 'name': r['name'], 'record_id': r['record_id'],
                        'split': r['split'], 'label': int(r['label']), 'weak_eye': r['weak_eye'],
                        'score': float(score), **{f'prediction_{p}': int(score >= cuts[p]) for p in POLICIES},
                    } for r, score in zip(records, scores)]
                    for policy in POLICIES:
                        for s in SPLITS:
                            rr = [p for p in predictions if p['split'] == s]
                            recomputed = metrics([p['label'] for p in rr], [p['score'] for p in rr], cuts[policy])
                            assert recomputed == result['metrics'][policy][s]
                    save(dest / 'predictions.json', predictions)
                    save(dest / 'results.json', result)
                    runs.append({'run_id': run_id, 'folder': str(dest), **result})
                    print('RESULT', run_id, json.dumps(result['metrics']['default']['test']), flush=True)
    assert len(runs) == 36 and sum(r['primary'] for r in runs) == 18
    save(output / 'summary.json', {'protocol': protocol, 'runs': runs})
    save(output / 'status.json', {'state': 'training_completed', 'experiments': len(runs), 'primary_experiments': 18, 'completed_at': datetime.now(timezone.utc).isoformat()})
    print('TRAINING COMPLETE', output, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=DEFAULT_OUT)
    parser.add_argument('--resume', action='store_true')
    arguments = parser.parse_args()
    with threadpool_limits(limits=4):
        run(arguments)
