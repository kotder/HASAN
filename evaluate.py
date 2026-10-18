import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf

from src.data import dataset_root, load_test
from src.metrics import evaluate_predictions, predict


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True, help='Model file or directory containing models')
    parser.add_argument('--dataset', choices=['gtsrb', 'chinese'], default='gtsrb')
    parser.add_argument('--data')
    parser.add_argument('--shared', action='store_true')
    parser.add_argument('--output', default='runs/evaluation')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--repeats', type=int, default=5)
    args = parser.parse_args()
    root = dataset_root(args.dataset, args.data)
    images, labels, frame = load_test(root, args.dataset, args.shared)
    checkpoint = Path(args.checkpoint)
    paths = sorted(checkpoint.glob('*.keras')) if checkpoint.is_dir() else [checkpoint]
    if checkpoint.is_dir() and not paths:
        paths = sorted(checkpoint.glob('*/last.keras'))
    if not paths or any(not p.is_file() for p in paths):
        raise FileNotFoundError('No checkpoints found')
    classes = range(43) if args.dataset == 'gtsrb' and not args.shared else np.unique(labels)
    rows = []
    for path in paths:
        model = tf.keras.models.load_model(path, compile=False)
        probabilities, timing = predict(model, images, args.batch_size, args.repeats)
        scores, per_class, confusion = evaluate_predictions(labels, probabilities, classes)
        name = path.parent.name if path.stem in {'last', 'best'} else path.stem
        out = Path(args.output) / name
        out.mkdir(parents=True, exist_ok=True)
        result = {'checkpoint': str(path), 'dataset': args.dataset, 'metrics': scores, 'inference': timing}
        (out / 'metrics.json').write_text(json.dumps(result, indent=2))
        per_class.to_csv(out / 'per_class.csv', index=False)
        frame['prediction'] = probabilities.argmax(axis=1)
        frame['confidence'] = probabilities.max(axis=1)
        frame['correct'] = frame['prediction'] == labels
        frame.to_csv(out / 'predictions.csv', index=False)
        np.save(out / 'confusion_matrix.npy', confusion)
        rows.append({'checkpoint': str(path), **scores, **timing})
        tf.keras.backend.clear_session()
    pd.DataFrame(rows).to_csv(Path(args.output) / 'summary.csv', index=False)
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == '__main__':
    main()
