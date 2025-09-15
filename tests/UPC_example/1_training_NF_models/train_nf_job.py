import os
import sys
import optax
import numpy as np
import random
import itertools

# Get the path three levels up and add it to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))

import src.train_nf as train_nf
import src.visualization_nf as visualize

seed = 42
np.random.seed(seed)
random.seed(seed)

# Grid search parameters
batch_sizes = [500, 1000, 2000, 5000]
layer_counts = [4, 6, 8, 10, 12]
learning_rates = [1e-3, 5e-4, 1e-4, 5e-5]

combinations = list(itertools.product(batch_sizes, layer_counts, learning_rates))

def generate_filename(prefix, batch_size, layers, lr):
    return f"{prefix}_bs{batch_size}_L{layers}_lr{lr:.0e}"

def main(index):
    # Ensure index is valid
    if index < 0 or index >= len(combinations):
        raise IndexError("Invalid combination index")

    batch_size, layers, lr = combinations[index]

    # Paths
    train_data_path = "data_eP/training_data.pkl"
    val_data_path = "data_eP/validation_data.pkl"
    model_path = f"models_eP/eP_nf_bs{batch_size}_L{layers}_lr{lr:.0e}.pkl"
    metrics_path = f"metrics_eP/eP_nf_bs{batch_size}_L{layers}_lr{lr:.0e}.dat"
    plot_path = f"plots_eP/eP_nf_bs{batch_size}_L{layers}_lr{lr:.0e}.png"

    # Training options
    train_nf.train(
        training_data_path=train_data_path,
        dimension=7,
        normalizing_flow_model_path=model_path,
        test_data_path=val_data_path,
        batch_size_training=batch_size,
        layers=layers,
        learning_rate=lr,
        number_training_steps=200000,
        seed=None,
        use_KL_divergence_loss=False,
        initial_normalizing_flow_model_path=None,
        loss_threshold_early_stopping=1e-5,
        optimizer=optax.adam,
        print_loss_training=True,
        metrics_mid_training=False,
        metrics_mid_training_frequency=500,
        metrics_path=metrics_path
    )

    visualize.visualize(
        model_path,
        train_data_path,
        80000,
        plot_path,
        seed,
        0.01
    )

if __name__ == "__main__":
    task_id = int(sys.argv[1])  # From SLURM_ARRAY_TASK_ID
    main(task_id)
