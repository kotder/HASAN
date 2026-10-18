import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image

from src.data import chinese_frame, dataset_root, gtsrb_frame


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', choices=['gtsrb', 'chinese'], default='gtsrb')
    parser.add_argument('--data')
    parser.add_argument('--output', default='runs/exploration')
    args = parser.parse_args()
    root = dataset_root(args.dataset, args.data)
    frame = chinese_frame(root) if args.dataset == 'chinese' else gtsrb_frame(root, 'Train')
    column = 'category' if args.dataset == 'chinese' else 'target'
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out / 'metadata.csv', index=False)
    counts = frame.groupby(column).size().rename('images')
    counts.to_csv(out / 'class_counts.csv')
    counts.plot.bar(figsize=(14, 4), ylabel='Images')
    plt.tight_layout()
    plt.savefig(out / 'class_counts.png', dpi=150)
    plt.close()
    samples = frame.groupby(column).sample(n=1, random_state=42)
    fig, axes = plt.subplots((len(samples) + 7) // 8, 8, figsize=(16, 2 * ((len(samples) + 7) // 8)))
    for ax in axes.flat:
        ax.axis('off')
    for ax, (_, row) in zip(axes.flat, samples.iterrows()):
        with Image.open(row['image_path']) as image:
            ax.imshow(image.convert('RGB'))
        ax.set_title(str(row[column]))
    fig.tight_layout()
    fig.savefig(out / 'samples.png', dpi=150)
    plt.close(fig)
    print(counts.to_string())


if __name__ == '__main__':
    main()
