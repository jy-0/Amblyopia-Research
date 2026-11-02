"""Run both released heads on the ten authorized anonymous image pairs."""

import csv
import json
from pathlib import Path

import torch
from core import encode_records, load_encoder, load_head, score_embeddings


def main():
    root = Path(__file__).resolve().parent
    data = root / 'sample_dataset'
    with (data / 'labels.csv').open(newline='', encoding='utf-8') as handle:
        rows = list(csv.DictReader(handle))
    records = [{f'{side}_image': data / r[f'{side}_image'] for side in ('right', 'left')}
               | {'right_crop': None, 'left_crop': None} for r in rows]
    torch.set_num_threads(4)
    encoder = load_encoder(root / 'weights/retfound_green.pth')
    eyes = encode_records(encoder, records)
    results = [{'patient_id': r['patient_id'], 'label': int(r['label'])} for r in rows]
    for name in ('logistic', 'rbf_svm'):
        head = load_head(root / f'weights/{name}_seed42.joblib')
        scores = score_embeddings(head, eyes)
        cut = head['metadata']['thresholds']['default']
        for record, score in zip(results, scores):
            record[name] = {'score': float(score), 'prediction': int(score >= cut)}
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
