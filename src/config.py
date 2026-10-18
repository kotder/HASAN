from copy import deepcopy

BASE_CONFIG = {
    'lr': 1e-3,
    'batch_size': 32,
    'epochs': 20,
    'optimizer': 'adam',
    'scheduler': 'none',
    'kernel_mode': 'mixed',
    'activation': 'relu',
}

def build_parameter_sweep(base_config):
    experiments = [('baseline', 'baseline', 'baseline', deepcopy(base_config))]

    def add(name, group, value, **overrides):
        config = deepcopy(base_config)
        config.update(overrides)
        experiments.append((name, group, str(value), config))

    for lr in [1e-2, 1e-3, 1e-4, 3e-5, 1e-5]:
        add(f'lr_{lr:g}', 'learning_rate', lr, lr=lr)
    for batch_size in [16, 32, 64, 128, 256]:
        add(f'batch_{batch_size}', 'batch_size', batch_size, batch_size=batch_size)
    for epochs in [20, 30, 40, 50, 60]:
        add(f'epochs_{epochs}', 'epochs', epochs, epochs=epochs)
    for optimizer in ['adam', 'adamw', 'sgd']:
        add(f'opt_{optimizer}', 'optimizer', optimizer, optimizer=optimizer)
    for scheduler in ['none', 'reducelronplateau', 'cosine']:
        add(f'sched_{scheduler}', 'scheduler', scheduler, scheduler=scheduler)
    for kernel_mode in ['all_5x5', 'all_3x3']:
        add(f'kernel_{kernel_mode}', 'kernel_mode', kernel_mode, kernel_mode=kernel_mode)
    for activation in ['relu', 'sigmoid', 'tanh']:
        add(f'act_{activation}', 'activation', activation, activation=activation)

    return experiments
