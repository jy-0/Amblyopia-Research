"""Offline prediction for one same-visit right/left raw-fundus pair."""
import argparse
import json
from pathlib import Path

import torch
from core import PairedFundusPredictor


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--classifier', choices=('logistic', 'rbf_svm'), required=True)
    parser.add_argument('--right', type=Path, required=True)
    parser.add_argument('--left', type=Path, required=True)
    parser.add_argument('--right-crop', nargs=4, type=int)
    parser.add_argument('--left-crop', nargs=4, type=int)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--threshold', choices=('default', 'validation_bacc', 'validation_sens90'), default='default')
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    torch.set_num_threads(4)
    model = PairedFundusPredictor(root / f'weights/{args.classifier}_seed42.joblib',
                                 root / 'weights/retfound_green.pth', args.device)
    result = model.predict_pair(args.right, args.left, args.right_crop, args.left_crop, args.threshold)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
