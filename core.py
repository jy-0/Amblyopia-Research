"""Shared frozen retinal encoder and independent patient-level image probes."""

import csv
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
from PIL import Image, ImageOps
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, confusion_matrix, f1_score, precision_score,
    roc_auc_score, roc_curve,
)
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import torch


FORMAT = 'paired_fundus_amblyopia_image_v1'
SPLITS = ('train', 'val', 'test')
C_GRID = (.001, .01, .1, 1., 10.)
GAMMA_GRID = ('scale', .0001, .001, .01)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def pair_features(eyes):
    eyes = np.asarray(eyes, dtype=np.float32)
    if eyes.ndim != 3 or eyes.shape[1:] != (2, 384) or not np.isfinite(eyes).all():
        raise ValueError('Expected finite [patients,2,384] embeddings in right,left order')
    right, left = eyes[:, 0], eyes[:, 1]
    return np.concatenate([(right + left) / 2, np.abs(right - left)], axis=1).astype(np.float64)


def preprocess(path, crop=None, size=392):
    with Image.open(path) as source:
        image = ImageOps.exif_transpose(source).convert('RGB')
        if crop is not None:
            if len(crop) != 4 or any(not isinstance(v, int) for v in crop):
                raise ValueError('Crop must be four integer pixel coordinates')
            x0, y0, x1, y1 = crop
            if not (0 <= x0 < x1 <= image.width and 0 <= y0 < y1 <= image.height):
                raise ValueError('Crop is outside the image')
            image = image.crop(crop)
        image = ImageOps.contain(image, (size, size), Image.Resampling.BILINEAR)
        canvas = Image.new('RGB', (size, size), 0)
        canvas.paste(image, ((size - image.width) // 2, (size - image.height) // 2))
        pixels = np.asarray(canvas).copy()
    return (torch.from_numpy(pixels).permute(2, 0, 1).float() / 255. - .5) / .5


def load_encoder(path, device='cpu', expected_sha256=None):
    import timm
    if expected_sha256 is not None and sha256(path) != expected_sha256:
        raise ValueError('Encoder checksum does not match the classifier')
    model = timm.create_model('vit_small_patch14_reg4_dinov2', img_size=(392, 392),
                              num_classes=0, pretrained=False)
    model.load_state_dict(torch.load(path, map_location='cpu', weights_only=True), strict=True)
    model.global_pool = 'avg'
    return model.requires_grad_(False).eval().to(device)


def read_manifest(path):
    path = Path(path)
    with path.open(encoding='utf-8-sig', newline='') as handle:
        reader = csv.DictReader(handle)
        required = {'patient_id', 'split', 'label', 'right_image', 'left_image'}
        if not required <= set(reader.fieldnames or []):
            raise ValueError(f'Manifest requires columns: {sorted(required)}')
        records = []
        for row in reader:
            if not any(v for v in row.values()):
                continue
            record = {'patient_id': row['patient_id'].strip(), 'split': row['split'].strip()}
            if not record['patient_id'] or record['split'] not in SPLITS or row['label'].strip() not in ('0', '1'):
                raise ValueError('Invalid patient ID, split or binary label')
            record['label'] = int(row['label'].strip())
            for side in ('right', 'left'):
                value = row[side + '_image'].strip()
                image = Path(value)
                record[side + '_image'] = str(image if image.is_absolute() else path.parent / image)
                crop = row.get(side + '_crop', '').strip()
                record[side + '_crop'] = json.loads(crop) if crop else None
            records.append(record)
    ids = [r['patient_id'] for r in records]
    if len(set(ids)) != len(ids):
        raise ValueError('One unique selected examination per patient is required; no cross-split patient overlap')
    for split in SPLITS:
        if {r['label'] for r in records if r['split'] == split} != {0, 1}:
            raise ValueError('Train, validation and test must each contain both classes')
    return records


@torch.inference_mode()
def encode_records(encoder, records, device='cpu', batch_size=4):
    output = []
    for start in range(0, len(records), batch_size):
        batch = records[start:start + batch_size]
        tensors = [preprocess(r[side + '_image'], r[side + '_crop']) for r in batch for side in ('right', 'left')]
        features = encoder(torch.stack(tensors).to(device)).cpu().numpy()
        output.append(features.reshape(len(batch), 2, 384))
    return np.concatenate(output).astype(np.float32)


def metrics(y, score, threshold):
    y, score = np.asarray(y, dtype=int), np.asarray(score, dtype=float)
    predicted = score >= threshold
    (tn, fp), (fn, tp) = confusion_matrix(y, predicted, labels=[0, 1])
    return {'patients': len(y), 'positive': int(y.sum()), 'negative': int(len(y) - y.sum()),
            'threshold': float(threshold), 'auc': float(roc_auc_score(y, score)),
            'accuracy': float(accuracy_score(y, predicted)),
            'sensitivity': float(tp / (tp + fn)), 'specificity': float(tn / (tn + fp)),
            'precision': float(precision_score(y, predicted, zero_division=0)),
            'f1': float(f1_score(y, predicted, zero_division=0)),
            'confusion_matrix': [[int(tn), int(fp)], [int(fn), int(tp)]]}


def validation_thresholds(y, score, default):
    fpr, tpr, values = roc_curve(y, score, drop_intermediate=False)
    valid = np.flatnonzero(np.isfinite(values))
    best = max(valid, key=lambda i: ((tpr[i] + 1 - fpr[i]) / 2, -abs(values[i] - default), -values[i]))
    high = max((i for i in valid if tpr[i] >= .9 - 1e-12), key=lambda i: (1 - fpr[i], values[i]))
    return {'default': float(default), 'validation_bacc': float(values[best]), 'validation_sens90': float(values[high])}


def fit_head(eyes, labels, splits, classifier='logistic', seed=42):
    if classifier not in ('logistic', 'rbf_svm'):
        raise ValueError('Classifier must be logistic or rbf_svm')
    x = pair_features(eyes)
    labels, splits = np.asarray(labels, dtype=int), np.asarray(splits)
    if len(x) != len(labels) or len(labels) != len(splits):
        raise ValueError('Feature, label and split lengths differ')
    if any(set(labels[splits == split]) != {0, 1} for split in SPLITS):
        raise ValueError('Each split requires both labels')
    train, val = splits == 'train', splits == 'val'
    scaler = StandardScaler().fit(x[train])
    z_train, z_val = scaler.transform(x[train]), scaler.transform(x[val])
    best_auc, best, trials = -np.inf, None, []
    for c in C_GRID:
        for gamma in (GAMMA_GRID if classifier == 'rbf_svm' else (None,)):
            if classifier == 'logistic':
                model = LogisticRegression(C=c, class_weight='balanced', solver='lbfgs',
                                           max_iter=3000, tol=1e-8, random_state=seed)
            else:
                model = SVC(C=c, gamma=gamma, kernel='rbf', class_weight='balanced', probability=False,
                            tol=1e-6, max_iter=100000, cache_size=512, random_state=seed)
            model.fit(z_train, labels[train])
            if (classifier == 'logistic' and model.n_iter_.max() >= 3000) or (classifier == 'rbf_svm' and model.fit_status_):
                raise RuntimeError('Classifier did not converge')
            score = model.predict_proba(z_val)[:, 1] if classifier == 'logistic' else model.decision_function(z_val)
            auc = float(roc_auc_score(labels[val], score))
            parameters = {'C': c, **({'kernel': 'rbf', 'gamma': gamma} if classifier == 'rbf_svm' else {})}
            trials.append({'parameters': parameters, 'validation_auc': auc})
            if auc > best_auc:
                best, best_auc, chosen = model, auc, parameters
    val_score = best.predict_proba(z_val)[:, 1] if classifier == 'logistic' else best.decision_function(z_val)
    cuts = validation_thresholds(labels[val], val_score, .5 if classifier == 'logistic' else 0.)
    artifact = {'format': FORMAT, 'scaler': scaler, 'classifier': best,
                'metadata': {'task': 'patient-level amblyopia', 'classifier': classifier, 'seed': seed,
                             'input_dim': 768, 'thresholds': cuts, 'parameters': chosen,
                             'class_mapping': {'0': 'non_amblyopia', '1': 'unilateral_amblyopia'},
                             'score_type': 'positive_probability' if classifier == 'logistic' else 'uncalibrated_margin'}}
    # Only now compute test scores: hyperparameters and thresholds are finalized.
    score = score_embeddings(artifact, eyes)
    result = {'parameters': chosen, 'thresholds': cuts, 'trials': trials,
              'metrics': {policy: {split: metrics(labels[splits == split], score[splits == split], cut)
                                  for split in SPLITS} for policy, cut in cuts.items()}}
    return artifact, result, score


def load_head(path):
    # joblib/pickle is executable serialization: load only trusted local releases.
    artifact = joblib.load(path)
    if artifact.get('format') != FORMAT:
        raise ValueError('Unsupported model format')
    if artifact['classifier'].classes_.tolist() != [0, 1] or artifact['classifier'].n_features_in_ != 768:
        raise ValueError('Unexpected label order or classifier dimensions')
    if artifact['scaler'].n_features_in_ != 768:
        raise ValueError('Unexpected scaler dimensions')
    return artifact


def score_embeddings(artifact, eyes):
    z = artifact['scaler'].transform(pair_features(eyes))
    model = artifact['classifier']
    if artifact['metadata']['classifier'] == 'logistic':
        return model.predict_proba(z)[:, 1]
    return model.decision_function(z)


class PairedFundusPredictor:
    def __init__(self, head_path, encoder_path, device='cpu'):
        self.head = load_head(head_path)
        self.device = device
        self.encoder = load_encoder(encoder_path, device, self.head['metadata'].get('encoder_sha256'))

    def predict_pair(self, right_image, left_image, right_crop=None, left_crop=None, policy='default'):
        if policy not in self.head['metadata']['thresholds']:
            raise ValueError('Unknown threshold policy')
        records = [{'right_image': right_image, 'left_image': left_image,
                    'right_crop': right_crop, 'left_crop': left_crop}]
        eyes = encode_records(self.encoder, records, self.device)
        score = float(score_embeddings(self.head, eyes)[0])
        threshold = self.head['metadata']['thresholds'][policy]
        prediction = int(score >= threshold)
        return {'task': self.head['metadata']['task'], 'classifier': self.head['metadata']['classifier'],
                'seed': self.head['metadata']['seed'], 'score': score,
                'probability': score if self.head['metadata']['classifier'] == 'logistic' else None,
                'score_type': self.head['metadata']['score_type'], 'threshold': threshold,
                'prediction': prediction, 'class_name': self.head['metadata']['class_mapping'][str(prediction)]}
