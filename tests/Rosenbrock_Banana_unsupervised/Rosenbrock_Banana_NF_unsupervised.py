import sys
import os

# Get the path two levels up and add it to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import optax
import src.train_nf as train_nf
import src.visualization_nf as visualize


# main function to run the training
if __name__ == "__main__":
    
    training_data_path = "training_data_Rosenbrock_Banana_unsupervised.pkl"
    dimension = 2
    normalizing_flow_model_path = "normalizing_flow_model_Rosenbrock_Banana_unsupervised.pkl"
    test_data_path = "validation_data_Rosenbrock_Banana_unsupervised.pkl"
    batch_size_training = 4000
    layers = 6
    learning_rate = 1e-3
    number_training_steps = 25000
    seed = 42
    use_KL_divergence_loss = True # Inactive for unsupervised training
    initial_normalizing_flow_model_path = None
    loss_threshold_early_stopping = 0
    optimizer = optax.adam
    print_loss_training = True
    metrics_mid_training = False
    metrics_mid_training_frequency = 1000
    metrics_path = 'metrics_Rosenbrock_Banana_unsupervised.dat'
    unsupervised_training = True
    train_nf.train(training_data_path, dimension, normalizing_flow_model_path, 
          test_data_path, batch_size_training, layers, learning_rate, 
          number_training_steps, seed, use_KL_divergence_loss, 
          initial_normalizing_flow_model_path, loss_threshold_early_stopping,
          optimizer, print_loss_training, metrics_mid_training,
          metrics_mid_training_frequency, metrics_path, unsupervised_training)
    # Visualize the training
    visualize.visualize(normalizing_flow_model_path, training_data_path, 15000, 
                         'corner_plot_Rosenbrock_Banana_unsupervised.png', 
                         seed, 0.04, unsupervised_training)
