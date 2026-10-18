import time

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, average_precision_score, confusion_matrix, recall_score


def evaluate_predictions(labels, probabilities, class_ids=None):
    predicted = probabilities.argmax(axis=1)
    classes = np.unique(labels) if class_ids is None else np.asarray(class_ids)
    per_class = []
    for c in classes:
        mask = labels == c
        per_class.append({'class_id': int(c), 'support': int(mask.sum()),
                          'recall': float(np.mean(predicted[mask] == c)) if mask.any() else 0.0,
                          'AP': float(average_precision_score(mask, probabilities[:, c])) if mask.any() else 0.0})
    table = pd.DataFrame(per_class)
    scores = {'images': len(labels), 'accuracy': float(accuracy_score(labels, predicted)),
              'recall_macro': float(table['recall'].mean()),
              'recall_micro': float(recall_score(labels, predicted, average='micro', zero_division=0)),
              'recall_weighted': float(recall_score(labels, predicted, average='weighted', zero_division=0)),
              'mAP': float(table['AP'].mean()),
              'cross_entropy': float(-np.log(np.clip(probabilities[np.arange(len(labels)), labels], 1e-12, 1)).mean())}
    for k in (3, 5):
        top = np.argsort(probabilities, axis=1)[:, -k:]
        scores[f'top{k}_accuracy'] = float(np.any(top == labels[:, None], axis=1).mean())
    return scores, table, confusion_matrix(labels, predicted, labels=np.arange(probabilities.shape[1]))


def predict(model, images, batch_size=32, repeats=5):
    if repeats < 1 or batch_size < 1:
        raise ValueError('Batch size and repeats must be positive')
    model.predict(images[:min(256, len(images))], batch_size=batch_size, verbose=0)
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        probabilities = model.predict(images, batch_size=batch_size, verbose=0)
        times.append(time.perf_counter() - start)
    mean = float(np.mean(times))
    return probabilities, {'batch_size': batch_size, 'repeats': repeats,
                           'mean_sec': mean, 'std_sec': float(np.std(times)),
                           'ms_per_image': 1000 * mean / len(images),
                           'images_per_sec': len(images) / mean}
