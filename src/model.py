import math
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPool2D
from tensorflow.keras.models import Sequential

def make_optimizer(name, learning_rate):
    name = name.lower()
    if name == 'adam':
        return tf.keras.optimizers.Adam(learning_rate=learning_rate)
    if name == 'adamw':
        return tf.keras.optimizers.AdamW(learning_rate=learning_rate)
    if name == 'sgd':
        return tf.keras.optimizers.SGD(learning_rate=learning_rate, momentum=0.9)
    raise ValueError(f'Unknown optimizer: {name}')

def make_scheduler(name, base_lr, epochs):
    name = name.lower()
    if name == 'none':
        return None
    if name == 'reducelronplateau':
        return tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss', factor=0.5, patience=3, min_lr=1e-7, verbose=0
        )
    if name == 'cosine':
        def cosine_lr(epoch, current_lr):
            progress = epoch / max(1, epochs - 1)
            return float(base_lr * 0.5 * (1.0 + math.cos(math.pi * progress)))
        return tf.keras.callbacks.LearningRateScheduler(cosine_lr, verbose=0)
    raise ValueError(f'Unknown scheduler: {name}')

def build_model(input_shape, num_classes, cfg):
    kernel_mode = cfg['kernel_mode']
    if kernel_mode == 'all_5x5':
        k1 = k2 = (5, 5)
    elif kernel_mode == 'all_3x3':
        k1 = k2 = (3, 3)
    elif kernel_mode == 'mixed':
        k1, k2 = (5, 5), (3, 3)
    else:
        raise ValueError(f'Unknown kernel mode: {kernel_mode}')

    activation = cfg['activation']
    model = Sequential([
        tf.keras.Input(shape=input_shape),
        Conv2D(32, kernel_size=k1, activation=activation),
        Conv2D(32, kernel_size=k1, activation=activation),
        MaxPool2D(pool_size=(2, 2)),
        Dropout(0.2),
        Conv2D(64, kernel_size=k2, activation=activation),
        Conv2D(64, kernel_size=k2, activation=activation),
        MaxPool2D(pool_size=(2, 2)),
        Dropout(0.2),
        Flatten(),
        Dense(256, activation=activation),
        Dropout(0.5),
        Dense(num_classes, activation='softmax'),
    ])
    model.compile(
        optimizer=make_optimizer(cfg['optimizer'], cfg['lr']),
        loss='categorical_crossentropy',
        metrics=['accuracy'],
    )
    return model
