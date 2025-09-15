import sys
import os
import threading
import signal

# Get the path two levels up and add it to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import optax
import src.train_nf as train_nf
import src.visualization_nf as visualize

def send_ctrl_c():
    print("⏰ Timeout reached (60s) — sending KeyboardInterrupt (Ctrl+C)")
    # Send SIGINT to the current process = behaves like Ctrl+C
    os.kill(os.getpid(), signal.SIGINT)

# main function to run the training
if __name__ == "__main__":
    # Launch exit timer
    timer = threading.Timer(45, send_ctrl_c)
    timer.start()

    try:
        training_data_path = "training_data_Rosenbrock_Banana.pkl"
        dimension = 2
        normalizing_flow_model_path = "normalizing_flow_model_Rosenbrock_Banana.pkl"
        test_data_path = "validation_data_Rosenbrock_Banana.pkl"
        batch_size_training = 4000
        layers = 6
        learning_rate = 1e-3
        number_training_steps = 25000
        seed = 42
        use_KL_divergence_loss = True
        initial_normalizing_flow_model_path = None
        loss_threshold_early_stopping = 0
        optimizer = optax.adam
        print_loss_training = True
        metrics_mid_training = True
        metrics_mid_training_frequency = 1000
        metrics_path = 'metrics_Rosenbrock_Banana.dat'
        train_nf.train(training_data_path, dimension, normalizing_flow_model_path, 
            test_data_path, batch_size_training, layers, learning_rate, 
            number_training_steps, seed, use_KL_divergence_loss, 
            initial_normalizing_flow_model_path, loss_threshold_early_stopping,
            optimizer, print_loss_training, metrics_mid_training,
            metrics_mid_training_frequency, metrics_path)
    except KeyboardInterrupt:
        print("Training interrupted by Ctrl+C (simulated) — NF state should be saved.")
    finally:
        timer.cancel()  # clean up if training finishes early
