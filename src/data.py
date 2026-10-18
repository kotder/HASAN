import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split


def dataset_root(dataset, path=None):
    if path:
        root = Path(path)
    else:
        import kagglehub
        slug = {'gtsrb': 'meowmeowmeowmeowmeow/gtsrb-german-traffic-sign',
                'chinese': 'dmitryyemelyanov/chinese-traffic-signs'}[dataset]
        root = Path(kagglehub.dataset_download(slug))
    filename = 'Train.csv' if dataset == 'gtsrb' else 'annotations.csv'
    if not (root / filename).exists():
        candidates = sorted(root.rglob(filename))
        if len(candidates) != 1:
            raise FileNotFoundError(f'Expected one {filename} under {root}')
        root = candidates[0].parent
    return root


def load_images(frame, crop=False):
    images = []
    for row in frame.itertuples(index=False):
        with Image.open(row.image_path) as image:
            image = image.convert('RGB')
            if crop:
                image = image.crop((int(row.x1), int(row.y1), int(row.x2), int(row.y2)))
            images.append(np.asarray(image.resize((32, 32)), dtype=np.uint8))
    if not images:
        raise ValueError('No images found')
    return np.asarray(images, dtype=np.float32) / 255.0


def gtsrb_frame(root, split):
    frame = pd.read_csv(Path(root) / f'{split}.csv')
    frame['image_path'] = frame['Path'].map(lambda p: str(Path(root) / p.replace('\\', '/')))
    frame['target'] = frame['ClassId'].astype(int)
    return frame


def load_training(root, seed=42):
    frame = gtsrb_frame(root, 'Train')
    images = load_images(frame)
    labels = frame['target'].to_numpy()
    return train_test_split(images, labels, test_size=0.2, stratify=labels, random_state=seed)


def chinese_frame(root):
    root = Path(root)
    frame = pd.read_csv(root / 'annotations.csv')
    sizes = {}
    for name in frame['file_name'].unique():
        with Image.open(root / 'images' / name) as image:
            sizes[name] = image.size
    valid = [sizes[r.file_name] == (r.width, r.height) for r in frame.itertuples()]
    frame = frame.loc[valid].copy()
    if frame['file_name'].duplicated().any():
        raise ValueError('Unresolved duplicate image annotations')
    frame['image_path'] = frame['file_name'].map(lambda n: str(root / 'images' / n))
    return frame.sort_values(['category', 'file_name']).reset_index(drop=True)


def load_test(root, dataset='gtsrb', shared=False):
    mapping = json.loads(Path(__file__).with_name('class_mapping.json').read_text())
    if dataset == 'chinese':
        frame = chinese_frame(root)
        frame = frame[frame['category'].astype(str).isin(mapping)].copy()
        frame['target'] = frame['category'].map(lambda c: mapping[str(c)]['target'])
    else:
        frame = gtsrb_frame(root, 'Test')
        if shared:
            frame = frame[frame['target'].isin([v['target'] for v in mapping.values()])].copy()
    frame = frame.reset_index(drop=True)
    return load_images(frame, crop=dataset == 'chinese'), frame['target'].to_numpy(), frame
