import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf

from src.config import BASE_CONFIG, build_parameter_sweep
from src.data import dataset_root, load_test, load_training
from src.metrics import evaluate_predictions, predict
from src.model import build_model, make_scheduler


class BestWeights(tf.keras.callbacks.Callback):
    def __init__(self):
        super().__init__()
        self.score = -np.inf
        self.weights = None

    def on_epoch_end(self, epoch, logs=None):
        score = (logs or {}).get('val_accuracy', -np.inf)
        if score > self.score:
            self.score = score
            self.weights = self.model.get_weights()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data')
    parser.add_argument('--output', default='runs')
    parser.add_argument('--config', help='JSON file overriding baseline settings')
    parser.add_argument('--sweep', action='store_true')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--force', action='store_true')
    args = parser.parse_args()
    config = dict(BASE_CONFIG)
    if args.config:
        overrides = json.loads(Path(args.config).read_text())
        if set(overrides) - set(config):
            raise ValueError('Unknown configuration keys')
        config.update(overrides)
    root = dataset_root('gtsrb', args.data)
    x_train, x_val, y_train, y_val = load_training(root, args.seed)
    x_test, y_test, _ = load_test(root)
    experiments = build_parameter_sweep(config) if args.sweep else [('baseline', 'baseline', 'baseline', config)]
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    for name, group, value, cfg in experiments:
        run = output / name
        run.mkdir(parents=True, exist_ok=True)
        result_path = run / 'metrics.json'
        if not args.force and all((run / f).exists() for f in ['metrics.json', 'last.keras', 'best.keras']):
            result = json.loads(result_path.read_text())
            if result['config'] != cfg or result['seed'] != args.seed or result['data'] != str(root.resolve()):
                raise ValueError(f'Run settings differ for {name}; use another output directory or --force')
        else:
            tf.keras.backend.clear_session()
            tf.keras.utils.set_random_seed(args.seed)
            tf.config.experimental.enable_op_determinism()
            model = build_model((32, 32, 3), 43, cfg)
            tracker = BestWeights()
            scheduler = make_scheduler(cfg['scheduler'], cfg['lr'], cfg['epochs'])
            callbacks = [tracker] + ([scheduler] if scheduler else [])
            start = time.perf_counter()
            history = model.fit(x_train, tf.keras.utils.to_categorical(y_train, 43),
                                validation_data=(x_val, tf.keras.utils.to_categorical(y_val, 43)),
                                batch_size=cfg['batch_size'], epochs=cfg['epochs'], callbacks=callbacks, verbose=2)
            elapsed = time.perf_counter() - start
            model.save(run / 'last.keras')
            probabilities, timing = predict(model, x_test)
            scores, per_class, confusion = evaluate_predictions(y_test, probabilities, range(43))
            result = {'experiment': name, 'group': group, 'value': value, 'config': cfg,
                      'seed': args.seed, 'data': str(root.resolve()), 'training_sec': elapsed,
                      'test': scores, 'inference': timing}
            per_class.to_csv(run / 'per_class.csv', index=False)
            np.save(run / 'confusion_matrix.npy', confusion)
            pd.DataFrame(history.history).to_csv(run / 'history.csv', index=False)
            if tracker.weights is None:
                raise RuntimeError('No validation checkpoint recorded')
            model.set_weights(tracker.weights)
            model.save(run / 'best.keras')
            temporary = result_path.with_suffix('.tmp')
            temporary.write_text(json.dumps(result, indent=2))
            temporary.replace(result_path)
        rows.append({'experiment': name, 'group': group, 'value': value,
                     'training_sec': result['training_sec'], **result['test'], **result['inference']})
        pd.DataFrame(rows).to_csv(output / 'summary.csv', index=False)
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == '__main__':
    main()
