# High Accuracy Sign Attention Network for Traffic Sign  Perception

## Setup

Python 3.10–3.12.

```bash
pip install -r requirements.txt
```

## Train

```bash
python train.py
python train.py --sweep
```

Datasets download automatically with KaggleHub. To use existing data, pass `--data /path/to/gtsrb` containing `Train.csv`, `Test.csv`, `Train/`, and `Test/`.

Use `--config config.json` to override training settings and `--output runs/custom` for a separate run. Completed runs are skipped; `--force` replaces them.

Default settings: Adam, learning rate 0.001, batch size 32, 20 epochs, ReLU, mixed 5×5/3×3 kernels, and seed 42. Inputs are RGB images resized to 32×32 and normalized to [0, 1]. Training uses a stratified 80/20 train-validation split.

The sweep runs 27 entries across learning rate, batch size, epochs, optimizer, scheduler, kernel size, and activation, changing one setting at a time. Some entries repeat the baseline settings.

## Evaluate

```bash
python evaluate.py --checkpoint runs/baseline/last.keras
python evaluate.py --checkpoint runs/baseline/last.keras --dataset chinese
python evaluate.py --checkpoint runs/baseline/last.keras --shared
```

For Chinese data, `--data` points to a directory containing `annotations.csv` and `images/`. Evaluation uses the 21 shared classes in `src/class_mapping.json`. Chinese images are cropped using their annotations. `--shared` restricts the German test set to those same classes.

To evaluate several models, pass a directory containing `.keras` files or the training output directory. Use separate output directories for different datasets:

```bash
python evaluate.py --checkpoint runs/baseline/last.keras --dataset chinese --output runs/chinese
```

Evaluation saves accuracy, recall, mAP, top-k accuracy, cross-entropy, inference timing, per-class metrics, predictions, and a confusion matrix. Metrics are stored as fractions. mAP is the mean of per-class average precision; macro recall averages class recalls equally.

Training saves `last.keras`, `best.keras`, `history.csv`, `metrics.json`, per-class results, and a confusion matrix under each run. `summary.csv` compares final-checkpoint results.

## Explore

```bash
python explore.py
python explore.py --dataset chinese --output runs/chinese_exploration
```

Exports metadata, class counts, and sample images. Chinese annotations are cleaned by matching recorded image dimensions to the image files.

## Layout

- `train.py`, `evaluate.py`, `explore.py`: commands
- `src/`: model, data loading, metrics, and sweep settings
- `configs/`: training configuration files
- `data/`: optional local datasets
- `runs/`: generated models and results

```bash
python train.py --config configs/baseline.json
```
