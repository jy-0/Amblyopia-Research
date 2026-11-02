"""Refit image probes on a locally authorized, preselected patient manifest."""
import argparse
import csv
import json
from pathlib import Path

import joblib
import numpy as np
import torch
from threadpoolctl import threadpool_limits

from core import encode_records, fit_head, load_encoder, read_manifest, sha256


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--classifier', choices=('logistic', 'rbf_svm'), required=True)
    parser.add_argument('--encoder', type=Path, default=Path(__file__).resolve().parent / 'weights/retfound_green.pth')
    parser.add_argument('--cache', type=Path, help='Optional private NPZ with eyes, patient_id, encoder_sha256')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--batch-size', type=int, default=4)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Use a new output directory; existing results are not overwritten')
    if args.batch_size <= 0:
        raise ValueError('Batch size must be positive')
    torch.set_num_threads(4)
    records = read_manifest(args.manifest)
    encoder_sha = sha256(args.encoder)
    if args.cache:
        with np.load(args.cache, allow_pickle=False) as data:
            ids = data['patient_id'].tolist()
            if len(set(ids)) != len(ids) or str(data['encoder_sha256'].item()) != encoder_sha:
                raise ValueError('Cache patient IDs or encoder checksum is invalid')
            index = {key: i for i, key in enumerate(ids)}
            eyes = data['eyes'][[index[r['patient_id']] for r in records]].copy()
    else:
        encoder = load_encoder(args.encoder, args.device)
        eyes = encode_records(encoder, records, args.device, args.batch_size)
    y = np.array([r['label'] for r in records])
    splits = np.array([r['split'] for r in records])
    with threadpool_limits(limits=4):
        artifact, result, scores = fit_head(eyes, y, splits, args.classifier, args.seed)
    artifact['metadata']['encoder_sha256'] = encoder_sha
    args.output.mkdir(parents=True)
    joblib.dump(artifact, args.output / 'classifier.joblib')
    (args.output / 'results.json').write_text(json.dumps(result, indent=2))
    np.savez_compressed(args.output / 'private_embeddings.npz', eyes=eyes,
                        patient_id=np.array([r['patient_id'] for r in records]), encoder_sha256=encoder_sha)
    with (args.output / 'private_predictions.csv').open('w', encoding='utf-8-sig', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(['patient_id', 'split', 'label', 'score', 'prediction_default'])
        for record, score in zip(records, scores):
            writer.writerow([record['patient_id'], record['split'], record['label'], float(score),
                             int(score >= artifact['metadata']['thresholds']['default'])])
    print(json.dumps(result['metrics']['default'], indent=2))


if __name__ == '__main__':
    main()
